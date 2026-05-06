# ONNX Runtime Reference - Chapters 22-30: Operators, Training, and Optimization

---

## 22. Operator Kernel System

### OpKernel Base Class
```cpp
class OpKernel {
public:
    explicit OpKernel(const OpKernelInfo& info);
    virtual ~OpKernel() = default;

    virtual Status Compute(OpKernelContext* context) const = 0;
    virtual Status ComputeAsync(OpKernelContext* context, Stream* stream);

    const OpKernelInfo& Info() const;
};
```

### OpKernelContext
```cpp
class OpKernelContext {
public:
    const OrtValue* GetInput(int index) const;
    OrtValue* GetOutput(int index, const TensorShape& shape);
    OrtValue* GetOutput(int index, const int64_t* dims, size_t dim_count);
    Status GetTempSpaceAllocator(AllocatorPtr* allocator);
    int NumInputs() const;
    int NumOutputs() const;
    Stream* GetComputeStream() const;
};
```

### OpKernelInfo
```cpp
class OpKernelInfo {
public:
    Status GetAttr(const std::string& name, int64_t* value) const;
    Status GetAttr(const std::string& name, float* value) const;
    Status GetAttr(const std::string& name, std::string* value) const;
    Status GetAttr(const std::string& name, std::vector<int64_t>* value) const;
    Status GetAttr(const std::string& name, std::vector<float>* value) const;
    const TensorShape GetInputShape(int index) const;
    ONNXTensorElementDataType GetInputType(int index) const;
    int GetInputCount() const;
    int GetOutputCount() const;
};
```

### Kernel Registration
```cpp
// KernelDefBuilder
KernelDefBuilder builder;
builder.SetName("MyOp")
       .SetDomain("my_domain")
       .SetProvider("CPUExecutionProvider")
       .TypeConstraint("T", DataTypeImpl::GetType<float>());

// Register with KernelRegistry
registry->Register(builder.Build(), [](const OpKernelInfo& info) -> OpKernel* {
    return new MyOpKernel(info);
});

// Macro-based registration
ONNX_OPERATOR_KERNEL_CLASS_NAME(kOnnxDomain, "MyOp", 1, MyOpKernel);
```

---

## 23. Shape Inference System

Shape inference propagates tensor shapes through the graph during model loading.

```cpp
// Shape inference for a graph
Status Graph::Resolve() {
    // InferShapes is called during Resolve()
    ORT_RETURN_IF_ERROR(InferShapes(*this, logger_));
}

// Custom shape inference function
OrtStatus* MyShapeInferFunc(const OrtShapeInferContext* ctx) {
    size_t num_inputs = 0;
    api.KernelContext_GetInputCount(ctx, &num_inputs);

    // Get input shape
    const OrtValue* input = nullptr;
    api.KernelContext_GetInput(ctx, 0, &input);

    // Set output shape same as input
    // ...
    return nullptr;
}
```

---

## 24. Custom Operator Registration

### C++ Custom Op
```cpp
struct MyOpKernel {
    MyOpKernel(const OrtApi& api, const OrtKernelInfo& info) {}
    void Compute(OrtKernelContext* context) { /* ... */ }
};

struct MyOp : Ort::CustomOpBase<MyOp, MyOpKernel> {
    const char* GetName() const { return "MyOp"; }
    const char* GetExecutionProviderType() const { return "CPU"; }
    size_t GetInputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
};

// Registration
Ort::CustomOpDomain domain("my_domain");
MyOp op;
domain.Add(&op);
opts.AddCustomOpDomain(std::move(domain));
```

### Python Custom Op
```python
import onnxruntime as ort

class MyCustomOp(ort.custom_op.CustomOp):
    def compute(self, *args):
        # Process inputs and return outputs
        return output

# Register via shared library
opts = ort.SessionOptions()
opts.register_custom_ops_library("my_ops.so")
```

---

## 25. Partitioning and Graph Splitting

The graph partitioner assigns nodes to execution providers:

1. EP.GetCapability() returns supported subgraphs
2. Nodes assigned to highest-priority EP that supports them
3. Unsupported nodes fall back to CPU EP
4. Data transfer nodes inserted between EP boundaries

---

## 26. Training API - ORTModule

```python
import onnxruntime.training.ortmodule as ortmodule

# Drop-in PyTorch replacement
from onnxruntime.training.ortmodule import ORTModule

model = MyPyTorchModel()
model = ORTModule(model)

# Use exactly like PyTorch
output = model(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
```

---

## 27. Quantization API

```python
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType

# Dynamic quantization (weights only)
quantize_dynamic(
    model_input="model.onnx",
    model_output="model_quant.onnx",
    weight_type=QuantType.QUInt8,  # or QuantType.QInt8
    per_channel=True,
    reduce_range=False,
    op_types_to_quantize=['MatMul', 'Gather']
)

# Static quantization (weights + activations)
from onnxruntime.quantization import CalibrationDataReader

class MyCalibrationReader(CalibrationDataReader):
    def get_next(self):
        # Return dict of input_name → numpy array
        return {"input": self.data[self.index]}

quantize_static(
    model_input="model.onnx",
    model_output="model_quant.onnx",
    calibration_data_reader=MyCalibrationReader(),
    quant_format=QuantFormat.QDQ,  # or QuantFormat.QOperator
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QUInt8,
    per_channel=True,
    nodes_to_exclude=[],
    extra_options={
        'ActivationSymmetric': True,
        'WeightSymmetric': True,
    }
)

# MatMulNBits quantization (4-bit)
from onnxruntime.quantization import matmul_4bits_quantizer

matmul_4bits_quantizer.quantize(
    model_input="model.onnx",
    model_output="model_4bit.onnx",
    block_size=128,
    is_symmetric=True,
    accuracy_level=4,
)
```

---

## 28. LoRA Adapter Support

```python
# Python API for LoRA adapters
sess = ort.InferenceSession("base_model.onnx")

# Add LoRA adapter
sess.add_lora_adapter("path/to/lora_adapter")

# Run inference with LoRA
results = sess.run(None, {"input": input_data})

# Remove LoRA adapter
sess.remove_lora_adapter("lora_adapter_name")
```

---

## 29. IO Binding and Advanced Inference

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("model.onnx",
    providers=[("CUDAExecutionProvider", {"device_id": 0})])

# Zero-copy GPU inference
io_binding = sess.io_binding()

# Bind input on GPU
input_arr = np.random.randn(1, 3, 224, 224).astype(np.float32)
ort_input = ort.OrtValue.ortvalue_from_numpy(input_arr, "cuda", 0)
io_binding.bind_ortvalue_input("input", ort_input)

# Bind output on GPU
io_binding.bind_output("output", "cuda", 0)

# Run without CPU↔GPU copies
sess.run_with_iobinding(io_binding)

# Get GPU output
gpu_output = io_binding.get_outputs()[0]
print(f"Output shape: {gpu_output.shape()}, device: {gpu_output.device_name()}")

# Copy to CPU only when needed
cpu_output = io_binding.copy_outputs_to_cpu()[0]
```

---

## 30. Auto Mixed Precision

```python
from onnxruntime.transformers import optimizer as ort_optimizer

# Convert model to FP16
optimized_model = ort_optimizer.optimize_model(
    "model.onnx",
    model_type='bert',
    num_heads=12,
    hidden_size=768,
    use_gpu=True,
    opt_level=1,
)

# Save FP16 model
optimized_model.convert_float_to_float16(
    keep_io_types=True,  # Keep input/output in FP32
    use_symbolic_shape_infer=True,
)
optimized_model.save_model_to_file("model_fp16.onnx")
```
