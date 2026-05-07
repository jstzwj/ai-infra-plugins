# bitsandbytes: Integration and ML Ecosystem

This document covers how bitsandbytes integrates with the broader machine learning ecosystem, including Hugging Face libraries, serving frameworks, and training toolkits. Each integration is described with its architecture, the specific bnb APIs it calls, and configuration patterns.

---

## Hugging Face Transformers

Transformers is the primary user-facing entry point for bitsandbytes quantization. The integration is deep, covering model loading, quantization, serialization, and optimizer management.

### BitsAndBytesConfig

The `BitsAndBytesConfig` dataclass in `transformers` (`utils/quantization_config.py`) is the user-facing configuration object. It maps directly to bnb constructor parameters:

| BitsAndBytesConfig Field | bnb Constructor Arg | Used By |
|---|---|---|
| `load_in_4bit` | Selects `bnb.nn.Linear4bit` | `replace_with_bnb_linear()` |
| `load_in_8bit` | Selects `bnb.nn.Linear8bitLt` | `replace_with_bnb_linear()` |
| `bnb_4bit_quant_type` | `quant_type` kwarg to `Linear4bit()` | 4-bit quantizer |
| `bnb_4bit_compute_dtype` | `compute_dtype` positional arg to `Linear4bit()` | 4-bit quantizer |
| `bnb_4bit_use_double_quant` | `compress_statistics` kwarg to `Linear4bit()` | 4-bit quantizer |
| `bnb_4bit_quant_storage` | `quant_storage` kwarg to `Linear4bit()` | 4-bit quantizer |
| `llm_int8_threshold` | `threshold` kwarg to `Linear8bitLt()` | 8-bit quantizer |
| `llm_int8_has_fp16_weight` | `has_fp16_weights` kwarg to `Linear8bitLt()` | 8-bit quantizer |
| `llm_int8_skip_modules` | Modules excluded from conversion | Both quantizers |
| `llm_int8_enable_fp32_cpu_offload` | Controls device_map filtering | Both quantizers |

Example usage:

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

# 4-bit NF4 with double quantization
config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.uint8,
)

# 8-bit with outlier threshold
config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=config_4bit,
    device_map="auto",
)
```

### from_pretrained() Integration

When loading a model with `from_pretrained()`, transformers orchestrates quantization through a layered architecture:

1. `BitsAndBytesConfig` is parsed from `quantization_config` parameter.
2. A quantizer class (`Bnb4BitHfQuantizer` or `Bnb8BitHfQuantizer`) is instantiated.
3. The quantizer calls `replace_with_bnb_linear()` to replace `nn.Linear` modules with `bnb.nn.Linear4bit` or `bnb.nn.Linear8bitLt`.
4. Weights are loaded and moved to GPU, triggering quantization (`.to(device)` on `Params4bit` or `Int8Params` triggers `_quantize()`).

For pre-quantized checkpoints, deserialization uses:
- `Params4bit.from_prequantized(data, quantized_stats, requires_grad, device, module)` for 4-bit.
- `Int8Params(data, requires_grad=False, **kwargs)` with `SCB` in kwargs for 8-bit.

### replace_with_bnb_linear()

This function in `transformers/integrations/bitsandbytes.py` performs model surgery:

```python
def replace_with_bnb_linear(model, modules_to_not_convert=None, quantization_config=None):
    """
    Replaces all nn.Linear layers in the model with either Linear4bit or Linear8bitLt,
    depending on the quantization_config.
    """
```

The function recursively walks the model's `named_children()`, skips modules listed in `llm_int8_skip_modules` (defaults to `["lm_head"]`), and constructs the appropriate bnb linear layer with the correct constructor arguments from the config.

### device_map="auto" with Quantized Models

When `device_map="auto"` is specified, transformers delegates to `accelerate` for device placement. The key behaviors are:

- Quantized layers are placed on GPU by default, with priority order: CUDA > NPU > HPU > XPU > CPU.
- CPU offloading is supported for 8-bit models with `llm_int8_enable_fp32_cpu_offload=True`.
- dtype casting is **blocked** on quantized models: calling `model.to(dtype=...)` raises `ValueError("You cannot cast a bitsandbytes model in a new dtype")`.
- Moving 8-bit models across devices requires bnb >= 0.48.0.

### Weight Serialization Format

Pre-quantized checkpoints on the HuggingFace Hub use these state_dict key patterns:

**4-bit checkpoint keys (per weight tensor):**
- `weight` -- The packed quantized data
- `weight.absmax` -- Absmax scales
- `weight.quant_map` -- Quantization code lookup table
- `weight.nested_absmax` -- Double-quantization absmax (if `use_double_quant=True`)
- `weight.nested_quant_map` -- Double-quantization code lookup
- `weight.quant_state.bitsandbytes__nf4` or `weight.quant_state.bitsandbytes__fp4` -- Packed quant state metadata

**8-bit checkpoint keys (per weight tensor):**
- `weight` -- The int8 quantized data
- `SCB` -- The scale column-wise absmax
- `weight_format` -- Format metadata (always `row` in current versions)

### Optimizer Integration

The Transformers `Trainer` registers bnb optimizer names and constructs them via `optim.AdamW`, `optim.Lion`, `optim.RMSprop`, `optim.AdEMAMix`, and their paged variants. Registered names include: `adamw_bnb`, `adamw_8bit`, `paged_adamw`, `paged_adamw_8bit`, `ademamix`, `ademamix_8bit`, `paged_ademamix`, `paged_ademamix_8bit`, `lion`, `lion_8bit`, `paged_lion`, `paged_lion_8bit`, `rmsprop_bnb`, `rmsprop_8bit`, `rmsprop_32bit`.

The `GlobalOptimManager.get_instance()` is called to register embedding layers for fp32 optimization when using 8-bit optimizers:

```python
manager = bnb.optim.GlobalOptimManager.get_instance()
manager.register_module_override(module, "weight", {"optim_bits": 32})
```

---

## PEFT (LoRA on Quantized Models)

PEFT (Parameter-Efficient Fine-Tuning) wraps bnb linear layer types with adapter-specific subclasses. This enables the QLoRA pattern: training LoRA adapters on top of 4-bit quantized base models.

### QLoRA Pattern

The canonical QLoRA workflow:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Step 1: Load 4-bit model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# Step 2: Prepare for training
model = prepare_model_for_kbit_training(model)

# Step 3: Add LoRA adapters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
```

### prepare_model_for_kbit_training()

This PEFT utility prepares a quantized model for training by:

1. Freezing the base model parameters.
2. Casting all parameters that are not `Params4bit` or `Int8Params` to `float32` for numerical stability.
3. Enabling gradient checkpointing if requested (reduces activation memory during backprop).
4. Ensuring the model's forward pass does not inadvertently cast inputs to incompatible dtypes.

### LoraConfig with target_modules

PEFT uses `isinstance` checks on the base layer to determine which bnb wrapper to use:

```python
if isinstance(target_base_layer, bnb.nn.Linear8bitLt):
    # dispatch to 8-bit LoRA wrapper
elif isinstance(target_base_layer, bnb.nn.Linear4bit):
    # dispatch to 4-bit LoRA wrapper
```

Each tuner method (LoRA, AdaLoRA, IA3, OFT, VeRA, RandLoRA, ROAD) has a dedicated `bnb.py` file containing specialized wrapper classes.

### Gradient Checkpointing Compatibility

Gradient checkpointing works with quantized models but requires care:

- When enabled, activations are recomputed during backprop rather than stored, saving memory at the cost of recomputation.
- The `Linear4bit` forward pass uses `MatMul4Bit.apply()` (an autograd function), which correctly handles gradient computation through the dequantize-then-matmul pattern.
- PEFT includes a defensive `result = result.clone()` after 4-bit base layer forward passes to work around a backprop issue with manipulated views on 4-bit linear output.

### Memory Savings: 4-bit Base + 16-bit LoRA

The memory breakdown for a typical QLoRA setup:

| Component | Precision | Memory (relative) |
|---|---|---|
| Base model weights | 4-bit (NF4) | ~0.5 bytes per parameter |
| LoRA adapter weights | 16-bit (bf16) | ~2 bytes per adapter parameter |
| Optimizer states | 32-bit (fp32) | ~4-8 bytes per trainable parameter |
| Activations | 16-bit (bf16) | Depends on sequence length |

For a 7B parameter model with LoRA rank 16 on all linear layers:
- 4-bit base: ~3.5 GB
- LoRA adapters: ~50 MB
- Paged Adam 8-bit optimizer states: ~100 MB
- Total training memory: ~6-8 GB (fits on a single consumer GPU)

### Merge/Unmerge Pattern

PEFT's merge/unmerge workflow is the most sensitive integration point. The 4-bit merge pattern:

```python
weight = self.get_base_layer().weight
kwargs = weight.__dict__
output = dequantize_bnb_weight(weight, state=weight.quant_state)  # calls bnb.functional.dequantize_4bit()
w_data = output + lora_delta
if "bnb_quantized" in kwargs:
    kwargs["bnb_quantized"] = False
kwargs["requires_grad"] = False
kwargs.pop("data", None)
self.get_base_layer().weight = bnb.nn.Params4bit(w_data.to("cpu"), **kwargs).to(weight.device)
```

The 8-bit merge pattern:

```python
weight = self.get_base_layer().weight
state = self.get_base_layer().state
output = dequantize_bnb_weight(weight, state=state)
w_data = output + lora_delta
self.get_base_layer().weight = bnb.nn.Int8Params(
    w_data.to("cpu"), requires_grad=False, has_fp16_weights=weight.has_fp16_weights
).to(weight.device)
state.reset_grads()
```

---

## Accelerate

Accelerate provides model loading, device placement, and offloading infrastructure for bnb-quantized models. The integration lives primarily in `accelerate/utils/bnb.py` and `accelerate/utils/modeling.py`.

### device_map Support for Quantized Models

Accelerate's `infer_auto_device_map()` handles quantized layers specially. When computing device maps, it accounts for the reduced memory footprint of quantized parameters:

- 4-bit parameters: counted at their actual storage size (packed 4-bit data + absmax + quant_state metadata).
- 8-bit parameters: counted at 1 byte per element (int8 weights) plus the SCB statistics tensor.

### Big Model Inference Protocol

Accelerate's `dispatch_model()` implements the "Big Model Inference" protocol, which enables loading models larger than GPU memory by:

1. Computing an optimal `device_map` that distributes model layers across available devices and CPU RAM.
2. Moving each layer to its assigned device just-in-time during forward passes.
3. Offloading layers back to CPU when not actively needed.

For bnb-quantized models, this protocol is particularly effective because the quantized weights are already small, allowing even 70B+ parameter models to run on limited hardware.

### CPU Offloading with Quantized Layers

Accelerate handles CPU offloading for 8-bit models with special logic in `set_module_tensor_to_device()`:

1. Quantized parameters are first moved to GPU to trigger quantization.
2. The resulting int8 weights and SCB statistics are then moved back to CPU.
3. During forward passes, the layer is temporarily moved to GPU, the computation is performed, and results are moved back.

This is controlled by `llm_int8_enable_fp32_cpu_offload=True` in `BitsAndBytesConfig`.

### dispatch_model() Integration

The `dispatch_model()` function uses string-based class name checks to identify bnb parameter types:

```python
param_cls.__name__ in ["Int8Params", "FP4Params", "Params4bit"]
module.__class__.__name__ == "Linear8bitLt"
module.__class__.__name__ == "Linear4bit"
```

This avoids requiring a direct bnb import but makes the integration sensitive to class renaming. The `"FP4Params"` check is for backward compatibility with older bnb versions.

### BnbQuantizationConfig

Accelerate defines its own `BnbQuantizationConfig` in `utils/dataclasses.py` that mirrors the transformers `BitsAndBytesConfig` fields:

| Field | Maps to |
|---|---|
| `load_in_8bit` | Use `bnb.nn.Linear8bitLt` |
| `load_in_4bit` | Use `bnb.nn.Linear4bit` |
| `llm_int8_threshold` | `threshold` kwarg to `Linear8bitLt` |
| `bnb_4bit_quant_type` | `quant_type` kwarg to `Linear4bit` |
| `bnb_4bit_use_double_quant` | `compress_statistics` kwarg to `Linear4bit` |
| `bnb_4bit_compute_dtype` | `compute_dtype` kwarg to `Linear4bit` |

### FSDP2 Compatibility

In `fsdp_utils.py`, accelerate checks for `Params4bit` by class name to disable `cpu_ram_efficient_loading` when 4-bit parameters are present, since FSDP2 cannot handle bnb parameter types during CPU-efficient loading.

---

## Diffusers

Diffusers integrates bitsandbytes for quantized inference of diffusion models (Stable Diffusion, etc.).

### Quantized UNet, VAE, and Text Encoders

The main components of a diffusion pipeline that benefit from quantization:

- **UNet**: The largest component; quantizing it to 8-bit or 4-bit significantly reduces memory.
- **Text Encoder** (CLIP/T5): Can be quantized to 4-bit for minimal quality impact.
- **VAE Decoder**: Typically left in full precision since it is small and quality-sensitive.

### 8-bit/4-bit Inference for Stable Diffusion Models

```python
from diffusers import StableDiffusionPipeline, BitsAndBytesConfig as DiffusersBnbConfig

# 4-bit NF4 quantization
pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    quantization_config=DiffusersBnbConfig(load_in_4bit=True),
    torch_dtype=torch.float16,
)

# 8-bit quantization
pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    quantization_config=DiffusersBnbConfig(load_in_8bit=True),
    torch_dtype=torch.float16,
)
```

### BitsAndBytesConfig in Pipeline.from_pretrained()

Diffusers' `from_pretrained()` accepts a `quantization_config` parameter that follows the same pattern as Transformers. The pipeline replaces `nn.Linear` modules in the UNet and text encoders with `bnb.nn.Linear4bit` or `bnb.nn.Linear8bitLt` during loading.

Key differences from Transformers integration:
- Only inference is supported (no training through Diffusers).
- The VAE decoder is typically excluded from quantization.
- The `device_map` parameter is handled by Diffusers internally.

---

## TGI (Text Generation Inference)

TGI is HuggingFace's production inference server. It does NOT use `bnb.nn.Linear8bitLt` or `bnb.nn.Linear4bit` directly. Instead, it builds custom wrapper modules around bnb primitives.

### Custom Linear Wrappers

TGI creates custom modules in `server/text_generation_server/layers/bnb.py`:

```
TGI layers/bnb.py:
  BNBWeight    -> wraps weight for 8-bit, calls own Linear8bitLt
  BNBFP4Weight -> wraps weight for fp4, calls own Linear4bit(quant_type="fp4")
  BNBNF4Weight -> wraps weight for nf4, calls own Linear4bit(quant_type="nf4")
  Linear8bitLt -> custom 8-bit linear using bnb.MatmulLtState + bnb.matmul()
  Linear4bit   -> custom 4-bit linear using bnb.nn.Params4bit + bnb.matmul_4bit()
```

### MatmulLtState Deprecated Fields

TGI accesses deprecated fields on `MatmulLtState` that are maintained for compatibility. In the bnb source code, these are defined as:

```python
# Deprecated attributes kept for downstream compatibility (TGI, vLLM).
# These are always None and will be fully removed in the next release.
_deprecated_fields = frozenset({"CxB", "CxBt", "formatB", "_tile_indices"})

def __getattr__(self, name):
    if name in MatmulLtState._deprecated_fields:
        warnings.warn(
            f"MatmulLtState.{name} is deprecated and will be removed in the next bitsandbytes release.",
            FutureWarning,
            stacklevel=2,
        )
        return None
```

TGI previously accessed `state.CxB` (turing/ampere format weights) and `state.formatB` after the first forward pass. In the current bnb version, the state management has been simplified and these fields always return `None` with a deprecation warning.

### Scaled Dot-Product Attention Integration

TGI integrates with PyTorch's `scaled_dot_product_attention` (SDPA) for efficient attention computation. When combined with bnb quantization, the attention weights are dequantized before being passed to SDPA:

```python
# Simplified TGI pattern
q = self.query_key_value(hidden_states)  # may be 4-bit/8-bit linear
# ... split into q, k, v ...
attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

### Continuous Batching with Quantized Models

TGI implements continuous batching (also called iterative scheduling), where multiple sequences are processed together with different prompt lengths and generation stages. With bnb quantization:

- The base model weights remain quantized and shared across all batch entries.
- The KV cache is stored in full precision (fp16/bf16).
- Dynamic batching does not affect the quantization state; the quantized weights are static.

---

## vLLM

vLLM is a high-throughput inference engine that integrates bnb for quantized model serving with PagedAttention.

### PagedAttention with Quantized Weights

vLLM's core innovation is PagedAttention, which manages the KV cache as virtual memory pages. When combined with bnb quantization:

- Model weights are stored in 4-bit (NF4/FP4) or 8-bit format.
- The KV cache is managed separately through PagedAttention and stored in full precision.
- During forward passes, weights are dequantized on-the-fly using `bnb.matmul_4bit()` or `bnb.matmul()`.

### Custom Kernel Integration

vLLM registers its own custom PyTorch operation for 4-bit matmul:

```python
torch.ops.vllm.apply_bnb_4bit = _apply_bnb_4bit
```

This wraps the per-shard matmul loop and is registered with a fake implementation for `torch.compile` compatibility. The actual computation calls `bnb.matmul_4bit()` internally.

### MatmulLtState Usage

vLLM constructs `MatmulLtState` per shard for 8-bit inference. Like TGI, it sets state attributes directly:

```python
matmul_state = bnb.MatmulLtState()
matmul_state.threshold = 0.0
matmul_state.has_fp16_weights = False
matmul_state.is_training = False
matmul_state.CB = weight.CB
matmul_state.SCB = weight.SCB
```

vLLM also accesses the deprecated `CxB` field on the state object, which returns `None` in the current version.

### QuantState Direct Usage

vLLM makes extensive use of `bitsandbytes.functional.QuantState`:

- `QuantState.from_dict(quant_state_dict, device=...)` -- Reconstructs QuantState from pre-quantized checkpoint keys.
- `QuantState(absmax=..., shape=..., code=..., blocksize=..., quant_type=..., dtype=...)` -- Constructs directly for MoE expert weight fusion.
- Direct access to attributes: `.absmax`, `.shape`, `.code`, `.blocksize`, `.dtype`, `.nested`, `.state2`, `.offset`.

### Double Quantization Optimization

vLLM dequantizes double-quantized (nested) absmax values at weight-loading time rather than inference time:

```python
if quant_state.nested:
    absmax = F.dequantize_blockwise(quant_state.absmax, quant_state.state2)
    absmax += quant_state.offset
    quant_state.absmax = absmax
    quant_state.nested = False
    # clear state2/offset to avoid re-dequantizing at inference time
```

This is a one-time cost during loading that saves repeated computation during inference.

### Weight Shard Management

vLLM implements tensor-parallel weight sharding for bnb models:

1. **Shard offsets** (`bnb_shard_offsets`) track where each shard begins/ends in the packed weight tensor.
2. **Per-shard quant states** (`bnb_quant_state`) maps shard index to `QuantState`.
3. **Per-shard matmul states** (`matmul_state`) is a list of `MatmulLtState` objects for 8-bit.
4. **Generation counter** (`generation`) tracks first vs subsequent forward passes.

---

## Other Integrations

### PyTorch Lightning

PyTorch Lightning works with bitsandbytes through the standard PyTorch optimizer interface. To use 8-bit optimizers:

```python
import lightning as L
import bitsandbytes as bnb

class LitModel(L.LightningModule):
    def configure_optimizers(self):
        return bnb.optim.AdamW8bit(self.parameters(), lr=1e-4)
```

No special integration code is needed; bnb optimizers are drop-in replacements for PyTorch optimizers. The `GlobalOptimManager` can be used to override optimizer precision for specific parameter groups.

### Lit-GPT

Lit-GPT (by Lightning AI) supports bnb quantization for inference. The integration pattern:

1. Load model weights in fp16/bf16.
2. Replace `nn.Linear` layers with `bnb.nn.Linear4bit` or `bnb.nn.Linear8bitLt`.
3. Call `model.to("cuda")` to trigger quantization.

Lit-GPT handles the replacement logic internally and provides CLI flags for selecting quantization type.

### Axolotl

Axolotl is a fine-tuning toolkit that uses bnb for QLoRA training. It configures quantization through YAML config files:

```yaml
# axolotl config
bf16: true
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_use_double_quant: true

# LoRA config
adapter: lora
lora_rank: 32
lora_alpha: 64
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
```

Axolotl translates these settings to `BitsAndBytesConfig` and `LoraConfig` under the hood.

### LLM-Training-Toolkit

Various LLM training toolkits (including FastChat, Open-Assistant, and custom training scripts) use bnb through the standard Transformers `Trainer` interface. The common pattern:

1. Create a `BitsAndBytesConfig` with 4-bit settings.
2. Load the model with `from_pretrained(quantization_config=...)`.
3. Apply PEFT adapters with `get_peft_model()`.
4. Train with `Trainer` or a custom training loop using bnb optimizers.

---

## bnb Module-Level Signals for Integrations

The bitsandbytes package exposes specific module-level attributes that integrations check:

### features dict

```python
# bitsandbytes/__init__.py
features = {"multi_backend"}
```

This signals to downstream libraries that the current bnb version supports multiple hardware backends (CUDA, ROCm, XPU, HPU, MPS, CPU).

### supported_torch_devices

```python
# bitsandbytes/__init__.py
supported_torch_devices = {
    "cpu",
    "cuda",   # NVIDIA/AMD GPU
    "xpu",    # Intel GPU
    "hpu",    # Intel Gaudi
    "npu",    # Ascend NPU
    "mps",    # Apple Silicon
}
```

Transformers checks this via `getattr(bnb, "supported_torch_devices", set())` in `validate_bnb_backend_availability()` to determine if the user's device is supported.

### Minimum Version Requirements

| Downstream Project | Minimum bnb Version | Context |
|---|---|---|
| Transformers | 0.46.1 | `BITSANDBYTES_MIN_VERSION` in `utils/import_utils.py` |
| PEFT | Any (feature-gated) | Checks `is_bnb_available()` and `is_bnb_4bit_available()` |
| Accelerate | 0.37.2 (8-bit), 0.39.0 (4-bit) | In `utils/imports.py` |
| vLLM | 0.46.1 | Checked in `BitsAndBytesLinearMethod` and `BitsAndBytesMoEMethod` |
