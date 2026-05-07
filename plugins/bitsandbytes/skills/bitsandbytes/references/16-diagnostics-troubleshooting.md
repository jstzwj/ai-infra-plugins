# bitsandbytes: Diagnostics and Troubleshooting

This document covers common errors, diagnostic tools, performance optimization tips, and frequently asked questions for bitsandbytes.

---

## Common Errors and Solutions

### 1. "CUDA kernel image not found" / "kernel image not present"

**Symptom:**
```
RuntimeError: CUDA kernel image not found for sm_XX
```
or
```
CUDA error: no kernel image is available for execution on the device
```

**Cause:** The compiled bitsandbytes binary does not include kernels for your GPU's SM (Streaming Multiprocessor) architecture. This typically occurs when:
- The pre-compiled binary was built for a different set of GPU architectures.
- Your GPU has a compute capability that was not included during compilation.
- There is a CUDA version mismatch between the binary and your driver.

**Fix:**
1. Check your GPU's compute capability:
   ```python
   import torch
   print(torch.cuda.get_device_capability())
   ```
2. Rebuild bitsandbytes from source to include your architecture:
   ```bash
   pip install bitsandbytes --no-binary :all:
   # or
   git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
   cd bitsandbytes
   pip install -e .
   ```
3. Ensure your CUDA toolkit version matches PyTorch's CUDA version.

---

### 2. "fatbinwrap" Errors

**Symptom:**
```
RuntimeError: Failed to load CUDA fatbinary
```
or
```
fatbinwrap: error while loading shared libraries
```

**Cause:** CUDA fat binary wrapping failure, typically caused by a mismatch between the CUDA toolkit used to compile bitsandbytes and the CUDA toolkit available at runtime.

**Fix:**
1. Verify CUDA toolkit matches:
   ```bash
   nvcc --version
   python -c "import torch; print(torch.version.cuda)"
   ```
2. Check `LD_LIBRARY_PATH`:
   ```bash
   echo $LD_LIBRARY_PATH
   # Should include the CUDA toolkit lib directory, e.g.:
   # /usr/local/cuda-12.1/lib64
   ```
3. If using conda, ensure the CUDA toolkit is installed in the conda environment:
   ```bash
   conda install cuda-toolkit=12.1
   ```

---

### 3. "library not loaded" / Version Mismatch

**Symptom:**
```
ImportError: libbitsandbytes_cuda118.so: cannot open shared object file: No such file or directory
```
or
```
CUDA VERSION MISMATCH
Requested CUDA version:          12.1
Detected PyTorch CUDA version:   11.8
Available pre-compiled versions: 11.8
```

**Cause:** The pre-compiled bitsandbytes binary was built for a different CUDA version than what PyTorch was compiled with.

**Fix:**
1. Install bitsandbytes with the correct CUDA version:
   ```bash
   pip install bitsandbytes
   # The installer should auto-detect the CUDA version from PyTorch
   ```
2. Override the CUDA version if auto-detection fails:
   ```bash
   BNB_CUDA_VERSION=121 pip install bitsandbytes
   ```
3. Or compile from source:
   ```bash
   git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
   cd bitsandbytes
   pip install -e .
   ```

The `BNB_CUDA_VERSION` and `BNB_ROCM_VERSION` environment variables can force loading a specific binary:
```bash
export BNB_CUDA_VERSION=121  # Force CUDA 12.1 binary
```

---

### 4. "Expected torch.int8 input tensors" in igemm

**Symptom:**
```
TypeError: Expected torch.int8 input tensors A and B, but got torch.float16 and torch.float16
```

**Cause:** The `int8_linear_matmul()` or `igemm()` function was called with tensors that are not `torch.int8`. This is a low-level function that requires pre-quantized inputs.

**Fix:** Ensure inputs are properly quantized before calling int8 matmul operations:
```python
# Correct: quantize first, then matmul
CA, SCA, outlier_cols = bnb.functional.int8_vectorwise_quant(A.to(torch.float16))
CB, SCB, _ = bnb.functional.int8_vectorwise_quant(B.to(torch.float16))
result = bnb.functional.int8_linear_matmul(CA, CB)

# Incorrect: passing float tensors directly
result = bnb.functional.int8_linear_matmul(A, B)  # TypeError!
```

For most use cases, you should use the higher-level `bnb.matmul()` or `bnb.nn.Linear8bitLt` which handle quantization internally.

---

### 5. "Linear4bit is torch.float16, but bnb_4bit_compute_dtype=torch.float32"

**Symptom:**
```
WARNING: Input type into Linear4bit is torch.float16, but bnb_4bit_compute_dtype=torch.float32 (default).
This will lead to slow inference.
```

**Cause:** Performance warning for the common case where fp16 inputs are combined with the default fp32 compute dtype. The dequantized 4-bit weights are cast to fp32, then the matmul result is cast back to fp16, adding unnecessary overhead.

**Fix:** Match the compute dtype to the input dtype:

```python
from transformers import BitsAndBytesConfig

# For fp16 inputs, use fp16 compute
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # Match input dtype
)

# For bf16 inputs (recommended), use bf16 compute
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # Match input dtype
)
```

This is set in `Linear4bit.set_compute_type()` in `bitsandbytes/nn/modules.py`. The warning is only emitted once per layer (controlled by `self.compute_type_is_set`).

---

### 6. "Embedding layer is not quantized. Please call .cuda() first"

**Symptom:**
```
RuntimeError: Embedding layer is not quantized. Please call .cuda() or .to(device) first.
```

**Cause:** The `Embedding8bit.forward()` method checks `hasattr(self.weight, "SCB")`. If the model has not been moved to GPU, the `Int8Params._quantize()` method has not been called and `SCB` is not set.

**Fix:** Call `model.to(device)` before running the forward pass:
```python
model = model.to("cuda")  # This triggers quantization of Int8Params
output = model(input_ids)
```

---

### 7. "FP4 quantization state not initialized"

**Symptom:**
```
WARNING: FP4 quantization state not initialized. Please call .cuda() or .to(device) on the LinearFP4 layer first.
```

**Cause:** The `fix_4bit_weight_quant_state_from_module()` function detected that the `Params4bit.quant_state` is `None` and the module-level `quant_state` is also `None`. This typically happens when:
- FSDP splits parameters and loses the quant_state reference.
- The model was serialized/deserialized in a way that detached the quant_state.

**Fix:**
1. For FSDP: Use the recovery mechanism in `fix_4bit_weight_quant_state_from_module()`:
   ```python
   # This is called automatically in Linear4bit.forward()
   # It recovers quant_state from the module's stored reference
   ```
2. Ensure `model.to(device)` is called before any forward pass.
3. For manual parameter manipulation, ensure `Params4bit.quant_state` is preserved.

The recovery function in `bitsandbytes/nn/modules.py`:
```python
def fix_4bit_weight_quant_state_from_module(module):
    if getattr(module.weight, "quant_state", None) is not None:
        return
    if getattr(module, "quant_state", None) is None:
        logger.warning("FP4 quantization state not initialized...")
    # Recover from module-level storage
    if not isinstance(module.weight, Params4bit):
        module.weight = Params4bit(module.weight, quant_storage=module.quant_storage, bnb_quantized=True)
    module.weight.quant_state = module.quant_state
```

---

### 8. Paged Optimizer Warnings on CPU

**Symptom:**
```
UserWarning: Paged optimizers are not supported on CPU. Falling back to non-paged optimizer behavior.
```

**Cause:** Paged optimizers (e.g., `PagedAdamW8bit`) use CUDA managed memory (`cudaMallocManaged`) for virtual memory paging. This feature is not available on CPU.

**Fix:**
1. Use non-paged variants when training on CPU:
   ```python
   optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
   # instead of bnb.optim.PagedAdamW8bit(...)
   ```
2. Or move the model and optimizer to GPU:
   ```python
   model = model.to("cuda")
   optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=1e-4)
   ```

The warning is emitted once per optimizer instance (controlled by `self._cpu_paged_warned` in `Optimizer8bit.get_state_buffer()`).

---

### 9. Block Size Divisibility Warnings

**Symptom:**
```
UserWarning: Some matrices hidden dimension is not a multiple of 64 and efficient inference kernels
are not supported for these (slow). Matrix input size found: torch.Size([1, 768])
```
or
```
UserWarning: Embedding size 30522 is not divisible by block size 64. This will lead to slow inference.
```

**Cause:** The efficient 4-bit inference kernel (`gemv_4bit`) requires the hidden dimension to be divisible by the blocksize. When it is not, the code falls back to the slower `MatMul4Bit.apply()` path that dequantizes the full weight matrix.

**Fix:**
1. Choose a blocksize that divides the hidden dimension. Valid blocksizes: 32, 64, 128, 256, 512, 1024, 2048, 4096.
2. For most transformer models with hidden_dim of 768, 1024, 2048, 4096, etc., the default blocksize of 64 works.
3. For embedding layers with vocab sizes that are not divisible, accept the performance penalty (embedding lookups are typically not the bottleneck).

---

## Diagnostics Module

bitsandbytes includes a built-in diagnostics system that can be invoked from the command line or programmatically.

### Command-Line Diagnostics

```bash
python -m bitsandbytes
```

This outputs:
- bitsandbytes version
- Platform and OS information
- Python version
- PyTorch version and CUDA/HIP/XPU version
- Related package versions (accelerate, diffusers, numpy, peft, transformers, etc.)
- CUDA compute capability and binary availability
- A sanity check that runs a quick optimizer step

Example output:
```
============================================================
bitsandbytes v0.50.0.dev0
============================================================
Platform: Linux-5.15.0-67-generic-x86_64-with-glibc2.35
  libc: glibc 2.35
Python: 3.11.9
PyTorch: 2.5.1
  CUDA: 12.1
  HIP: N/A
  XPU: N/A
Related packages:
  accelerate: 1.2.1
  diffusers: 0.32.2
  numpy: 1.26.4
  peft: 0.14.0
  safetensors: 0.4.5
  transformers: 4.47.1
  triton: 3.1.0

PyTorch settings found: CUDA_VERSION=121, Highest Compute Capability: (8, 6).
Checking that the library is importable and CUDA is callable...
SUCCESS!
```

### Programmatic Diagnostics

```python
from bitsandbytes.diagnostics.main import show_environment, sanity_check
from bitsandbytes.cuda_specs import get_cuda_specs

# Show environment info
show_environment()

# Get CUDA specs
cuda_specs = get_cuda_specs()
if cuda_specs:
    print(f"CUDA version: {cuda_specs.cuda_version_string}")
    print(f"Compute capability: {cuda_specs.highest_compute_capability}")
    print(f"Has IMMA (tensor cores): {cuda_specs.has_imma}")
```

### CUDA Device Detection and Compute Capability Check

The `cuda_specs.py` module provides:

```python
from bitsandbytes.cuda_specs import (
    get_compute_capabilities,   # List of (major, minor) for all GPUs
    get_cuda_version_tuple,     # (major, minor) e.g., (12, 1)
    get_cuda_version_string,    # "121"
    get_cuda_specs,             # CUDASpecs dataclass
    get_rocm_gpu_arch,          # ROCm GPU architecture string
)

# Check tensor core support
specs = get_cuda_specs()
if specs.has_imma:
    print("GPU supports INT8 tensor cores (SM 7.5+)")
else:
    print("WARNING: Only slow 8-bit matmul is supported for your GPU!")
```

The `CUDASpecs` dataclass:
```python
@dataclasses.dataclass(frozen=True)
class CUDASpecs:
    highest_compute_capability: tuple[int, int]
    cuda_version_string: str
    cuda_version_tuple: tuple[int, int]

    @property
    def has_imma(self) -> bool:
        return torch.version.hip or self.highest_compute_capability >= (7, 5)
```

### Memory Info Reporting

bitsandbytes does not directly report GPU memory, but the diagnostics module includes checks for binary availability. You can combine it with PyTorch memory reporting:

```python
import torch

print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"GPU max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

---

## Performance Tips

### Use NF4 over FP4 for Normally-Distributed Weights

NF4 (NormalFloat4) is information-theoretically optimal for data that follows a normal distribution. Since neural network weights are approximately normally distributed, NF4 almost always produces better accuracy than FP4:

```python
# Recommended
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4 - optimal for weights
)

# Not recommended unless you have a specific reason
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="fp4",  # Floating point 4-bit - less optimal
)
```

### Enable Double Quantization (compress_statistics=True)

Double quantization compresses the absmax scaling factors themselves, saving additional memory:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Compress absmax scaling factors
)
```

Memory savings per parameter:
- Without double quant: 4 bits (weight) + 32 bits / blocksize (absmax) = ~4.5 bits/elem with blocksize=64
- With double quant: 4 bits (weight) + ~0.127 bits (quantized absmax) = ~4.127 bits/elem

### Use gemv_4bit for Single-Batch Inference (Automatic)

The `gemv_4bit` kernel is optimized for single-batch inference (vector-matrix multiplication). bitsandbytes automatically selects this path when the input is a vector:

```python
# In matmul_4bit() - automatic detection
if A.numel() == A.shape[-1] and A.requires_grad == False and A.device.type != "hpu":
    # Use fast gemv_4bit kernel
    out = F.gemv_4bit(A, B.t(), out, state=quant_state)
else:
    # Fall back to dequantize + standard matmul
    return MatMul4Bit.apply(A, B, out, bias, quant_state)
```

This is automatic; no user configuration needed.

### Set compute_dtype to Match Input dtype

The `compute_dtype` controls the precision of the matmul after dequantization. Mismatched dtypes cause unnecessary casting:

```python
# For bf16 inputs (recommended for modern GPUs)
bnb_4bit_compute_dtype=torch.bfloat16

# For fp16 inputs
bnb_4bit_compute_dtype=torch.float16

# For fp32 inputs
bnb_4bit_compute_dtype=torch.float32
```

The `Linear4bit.set_compute_type()` method automatically detects the input dtype and sets `compute_dtype` accordingly for bf32 and bf16 inputs. For fp16 inputs, it emits the performance warning described in error #5 above.

### Use bfloat16 When Available for Stability

bfloat16 has the same dynamic range as float32 (8 exponent bits) but reduced mantissa precision (7 bits). This makes it more numerically stable than float16 (5 exponent bits, 10 mantissa bits) for quantized operations:

```python
# Preferred for A100, H100, RTX 30xx, RTX 40xx, etc.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # More stable than float16
    quantization_config=bnb_config,
)
```

bfloat16 is supported on:
- NVIDIA: SM 8.0+ (Ampere and later)
- AMD: ROCm 5.0+ (MI200 and later)
- Intel: XPU (Arc, Data Center GPU Max)

### min_8bit_size Tuning

The `min_8bit_size` parameter in 8-bit optimizers controls the minimum parameter tensor size for 8-bit quantization of optimizer states. Parameters with fewer elements use 32-bit states:

```python
optimizer = bnb.optim.AdamW8bit(
    model.parameters(),
    lr=1e-4,
    min_8bit_size=4096,  # Default: only quantize params with > 4096 elements
)
```

Tuning guidelines:
- Small parameters (< 4096 elements): 8-bit quantization overhead outweighs memory savings. Keep at 32-bit.
- Large parameters (> 4096 elements): 8-bit quantization provides significant memory savings.
- For very large models, you can increase this threshold to skip medium-sized parameters that might be sensitive to quantization noise.

### Paged Optimizers for Models > 1B Parameters

Paged optimizers use CUDA unified memory to page optimizer states between GPU and CPU memory:

```python
# Paged optimizer - pages states to CPU when GPU memory is scarce
optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=1e-4)
```

The paging mechanism uses `GlobalPageManager` singleton to manage paged tensors. Prefetching brings states back to GPU before they are needed:

```python
# Automatic prefetching happens in Optimizer8bit.prefetch_state()
if self.is_paged:
    F.prefetch_tensor(state["state1"])
    if "state2" in state:
        F.prefetch_tensor(state["state2"])
```

Paged optimizers allocate managed memory via `cget_managed_ptr()` for parameters with more than 1 million elements (approximately 1 MB per state buffer).

---

## FAQs

### GPU Requirement

**Q: What GPU do I need?**

**A:**
- **Minimum**: SM 6.0 (Pascal, e.g., GTX 1080, P100). Only slow 8-bit matmul paths are available.
- **Recommended**: SM 7.5+ (Turing, Ampere, Hopper, Blackwell). Full INT8 tensor core support for fast 8-bit matmul. All features supported.
- **ROCm**: Fully supported from ROCm 6.1+.
- **Apple Silicon**: MPS backend supported with limitations.
- **Intel GPU**: XPU backend supported.
- **Intel Gaudi**: HPU backend supported.

The diagnostics module checks this:
```python
if not cuda_specs.has_imma:
    print("WARNING: Compute capability < 7.5 detected! Only slow 8-bit matmul is supported!")
```

### Training vs Inference

**Q: Should I use 4-bit or 8-bit quantization?**

**A:**
- **4-bit (QLoRA) for training**: Use NF4 with LoRA adapters. The base model stays in 4-bit while LoRA adapters are trained in bf16/fp16. This is the standard approach for fine-tuning LLMs on consumer hardware.
- **8-bit (LLM.int8()) for inference**: Use `Linear8bitLt` for inference-only deployments. Provides better accuracy than 4-bit at the cost of higher memory usage. The mixed-precision decomposition (threshold > 0) handles outlier features for near-lossless quality.
- **4-bit for inference**: Also works well, especially with NF4. Lower memory than 8-bit with slightly reduced accuracy.

### Multiple GPU Support

**Q: Does bitsandbytes work with multiple GPUs?**

**A:** Yes, through `device_map="auto"` (via Accelerate). Key behaviors:
- The model is automatically sharded across available GPUs.
- Each GPU holds a subset of the quantized layers.
- CPU offloading is available for additional capacity.
- The `features = {"multi_backend"}` flag in `bitsandbytes/__init__.py` signals that multi-backend support is available.

When multiple GPUs are present, bitsandbytes uses a device context manager to switch to the correct device before invoking CUDA kernels:
```python
# In functional.py
if torch.cuda.device_count() > 1:
    def _cuda_device_of(a):
        return torch.cuda.device_of(a)
else:
    # No overhead for single-GPU setups
    def _cuda_device_of(a):
        return contextlib.nullcontext()
```

### Gradient Checkpointing Compatibility

**Q: Can I use gradient checkpointing with quantized models?**

**A:** Yes. Gradient checkpointing works with both 4-bit and 8-bit quantized models:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    device_map="auto",
)
model.gradient_checkpointing_enable()
```

Notes:
- Gradient checkpointing re-runs the forward pass for each layer during backprop, which includes re-dequantizing 4-bit weights. This adds computation but saves activation memory.
- The `MatMul4Bit` and `MatMul8bitLt` autograd functions correctly handle this pattern.
- PEFT's `prepare_model_for_kbit_training()` can enable gradient checkpointing automatically.

### torch.compile Support via register_fake()

**Q: Does bitsandbytes work with torch.compile?**

**A:** Yes, bitsandbytes registers PyTorch custom ops with fake implementations for `torch.compile` compatibility. This is implemented in `bitsandbytes/_ops.py`:

```python
# Each custom op has a register_fake() implementation
if hasattr(torch.library, "register_fake"):
    register_fake = torch.library.register_fake
    register_kernel = torch.library.register_kernel
else:
    # PyTorch <= 2.3
    register_fake = torch.library.impl_abstract
    register_kernel = torch.library.impl
```

The following custom ops are defined with fake implementations:
- `bitsandbytes::int8_mixed_scaled_mm`
- `bitsandbytes::int8_scaled_mm`
- `bitsandbytes::int8_linear_matmul`
- `bitsandbytes::int8_vectorwise_quant`
- `bitsandbytes::int8_vectorwise_dequant`
- `bitsandbytes::int8_mm_dequant`
- `bitsandbytes::int8_double_quant`
- `bitsandbytes::dequantize_4bit`
- `bitsandbytes::quantize_4bit`
- `bitsandbytes::dequantize_blockwise`
- `bitsandbytes::quantize_blockwise`
- `bitsandbytes::gemv_4bit`
- `bitsandbytes::optimizer_update_32bit`
- `bitsandbytes::optimizer_update_8bit_blockwise`

Additionally, `Params4bit` uses `@property` decorators instead of `__getattr__` for FSDP state_dict traversal attributes, because Dynamo can trace descriptor protocol access but not `__getattr__` on Tensor subclasses.

### Error Handler Mock Library

When the native library fails to load, bitsandbytes provides a detailed error handler:

```python
class ErrorHandlerMockBNBNativeLibrary(BNBNativeLibrary):
    """Mock library handler that defers errors until native methods are called.
    
    Error scenarios covered:
    1. Missing shared library dependencies
    2. CUDA version mismatch
    3. Missing pre-compiled binaries
    4. Custom BNB_CUDA_VERSION/BNB_ROCM_VERSION override mismatches
    5. CPU-only installation attempts
    """
```

This mock generates formatted error messages with troubleshooting guidance, including available binary versions, compile-from-source instructions, and a link to create GitHub issues.

### Environment Variables

| Variable | Purpose |
|---|---|
| `BNB_CUDA_VERSION` | Override CUDA version for binary selection (e.g., `121` for CUDA 12.1) |
| `BNB_ROCM_VERSION` | Override ROCm version for binary selection (e.g., `72` for ROCm 7.2) |
| `LD_LIBRARY_PATH` | Must include CUDA runtime library path |

The binary loading logic in `cextension.py`:
```python
cuda_override_value = os.environ.get("BNB_CUDA_VERSION")
rocm_override_value = os.environ.get("BNB_ROCM_VERSION")
```

If `BNB_CUDA_VERSION` is set on a ROCm build (or vice versa), a `RuntimeError` is raised with guidance to use the correct variable.
