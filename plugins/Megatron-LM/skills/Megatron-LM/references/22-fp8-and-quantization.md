# Chapter 22: FP8 and Quantization

## Source Files
- `megatron/core/fp8_utils.py` - FP8 utility functions, recipe selection, context management
- `megatron/core/quantization/quant_config.py` - Quantization configuration framework
- `megatron/core/quantization/utils.py` - Quantization helper functions
- `megatron/core/transformer/transformer_config.py` - FP8 config fields in TransformerConfig
- `megatron/core/enums.py` - Fp8Recipe, Fp4Recipe enums

## FP8 Formats

Megatron-LM supports two FP8 data formats through Transformer Engine:

### E4M3 (4-bit exponent, 3-bit mantissa)
- Range: approximately [+/- 448, 0]
- No inf/NaN representation
- Used primarily for forward pass GEMM inputs and weights
- Better precision for values near zero

```bash
--fp8 e4m3
```

### Hybrid (E4M3 + E5M2)
- E4M3 for forward pass (inputs, weights)
- E5M2 (5-bit exponent, 2-bit mantissa) for backward pass gradients
- E5M2 has wider dynamic range but lower precision
- This is the recommended format for most training workloads

```bash
--fp8 hybrid
```

## FP8 Recipes

The FP8 recipe controls how scaling factors are computed and applied. Recipes are selected via `--fp8-recipe` and require specific Transformer Engine versions.

### Delayed Scaling

The original FP8 recipe. Scaling factors are computed from the previous iteration's maximum absolute value (amax).

```bash
--fp8 hybrid
--fp8-recipe delayed
--fp8-wgrad                  # Also quantize weight gradients
```

**How it works:**
1. During forward, amax values are recorded for each tensor
2. Scaling factors for the next iteration are computed from the current amax history
3. The "delay" means scaling factors lag one step behind
4. Amax history is reduced across the amax reduction group

**Configuration options:**
- `--fp8-wgrad`: Enable FP8 for weight gradients (backward dgrad is always FP8 in hybrid mode; wgrad is optional)
- `--fp8-amax-compute-algo most_recent`: Use only the most recent amax
- `--fp8-amax-compute-algo max`: Use max over amax history window

### Tensorwise Current Scaling

Computes per-tensor scaling factors from the current tensor's values rather than history.

```bash
--fp8 hybrid
--fp8-recipe tensorwise     # Requires TE >= 2.2.0
--fp8-dot-product-attention # Optional: FP8 for attention
```

Advantages over delayed scaling:
- No lag in scaling factor computation
- Potentially better accuracy for rapidly changing distributions
- No amax history buffer needed

### MXFP8 (Microscaling FP8)

Block-wise scaling where tensors are divided into 32-element blocks, each with its own scaling factor.

```bash
--fp8 hybrid
--fp8-recipe mxfp8                      # Requires TE >= 2.1.0
--fp8-param-gather                      # FP8 all-gather for params
--reuse-grad-buf-for-mxfp8-param-ag     # Reuse grad buffer for param AG
```

**Key characteristics:**
- Alignment size: 32 elements per block
- Per-block scaling provides better accuracy for tensors with varying magnitudes
- Requires `--fp8-param-gather` for parameter all-gathering
- The `--reuse-grad-buf-for-mxfp8-param-ag` option reduces memory by reusing the gradient buffer for parameter all-gathering
- Without `--reuse-grad-buf-for-mxfp8-param-ag`, significant additional GPU memory is used

### Blockwise Scaling

Extended block-wise FP8 with larger block sizes.

```bash
--fp8 hybrid
--fp8-recipe blockwise     # Requires TE >= 2.3.0
```

Uses `Float8BlockScaling` from Transformer Engine for finer-grained quantization control.

### Custom Recipe

Allows plugging in a custom quantizer factory:

```bash
--fp8 hybrid
--fp8-recipe custom
--fp8-quantizer-factory my_package.my_module.my_quantizer
```

The quantizer factory must be a callable that returns a Transformer Engine `CustomRecipe` object. Requires TE >= 2.9.0.

## FP8 with Distributed Optimizer

FP8 can be combined with the distributed optimizer for maximum memory savings:

```bash
--fp8 hybrid --fp8-recipe mxfp8
--use-distributed-optimizer
--fp8-param-gather
```

### FP8 Param Gather

With `--fp8-param-gather`, parameters are stored in FP8 in the model but maintained in FP32 in the optimizer's main parameters. During all-gather, FP8 parameters are gathered directly, reducing communication volume by 2x compared to BF16.

The conversion flow:
1. FP32 main params -> quantize to FP8 model params (done in `quantize_param_shard`)
2. FP8 model params are all-gathered across DP ranks
3. FP8 params are dequantized on-the-fly during forward/backward via Transformer Engine

### FP8 Tensor Utilities

Key functions in `fp8_utils.py`:

| Function | Description |
|----------|-------------|
| `is_float8tensor(tensor)` | Check if tensor is a TE FP8 tensor |
| `is_mxfp8tensor(tensor)` | Check if tensor is MXFP8 |
| `dequantize_fp8_tensor(tensor)` | Dequantize FP8 to higher precision |
| `modify_underlying_storage(tensor, new_data)` | Replace FP8 tensor's raw data |
| `quantize_param_shard(...)` | Cast FP32 main params to FP8 |
| `correct_amax_history_if_needed(model)` | Fix amax history for TE 1.x |
| `post_all_gather_processing(params)` | Post all-gather processing |
| `get_fp8_recipe(config)` | Get TE FP8 recipe from config |
| `get_fp8_context(config, layer_no)` | Get FP8 context manager |

### TE Version Compatibility

Different TE versions have different FP8 tensor implementations:

| TE Version | FP8 Tensor Class | Notes |
|------------|------------------|-------|
| 1.x | `Float8Tensor` | Uses `_fp8_meta` dict, `_fp8_meta_index` |
| 2.0 | `Float8Tensor` (from QuantizedTensor) | Transition version |
| 2.2+ | `QuantizedTensor` | Unified base class, `replace_raw_data` available |
| 2.3+ | `QuantizedTensor` | `cast_master_weights_to_fp8` supports FSDP |

## FP4 Format (NVFP4)

Megatron-LM also supports NVFP4 quantization through the quantization config framework:

```bash
--quantization-recipe path/to/nvfp4_recipe.yaml
```

FP4 uses 4-bit floating point representation for even more aggressive compression. The `Fp4Recipe` enum defines supported FP4 recipes.

## Quantization Configuration Framework

The quantization framework in `megatron/core/quantization/` provides a YAML-based system for per-layer quantization configuration.

### YAML Recipe Format

```yaml
configs:
  nvfp4:
    quant_cfg:
      "*weight_quantizer":
        num_bits: [4, 3]
        axis: null
      "*input_quantizer":
        enable: false
      "*output_layer*":
        enable: false
      "default":
        enable: false
    algorithm: max

  mxfp8:
    quant_cfg:
      "*weight_quantizer":
        num_bits: [4, 3]
        block_sizes: {-1: 128, -2: 128}
      "*input_quantizer":
        enable: false
      "default":
        enable: false
    algorithm: max

matchers:
  fc1:
    config: "nvfp4"
    type: "glob"
    pattern: "*fc1*"
    enabled: true
  fc2:
    config: "nvfp4"
    type: "glob"
    pattern: "*fc2*"
    enabled: true
  default:
    config: "mxfp8"
    type: "glob"
    pattern: "*"
    enabled: true
```

### Components

**Configs section:** Defines named quantization configurations. Each config is a dictionary consumed by the operator's quantization logic.

**Matchers section:** Ordered list of pattern matchers. The first match wins:

| Matcher Type | Description |
|-------------|-------------|
| `glob` | Bash-style glob matching on module path |

Each matcher specifies:
- `config`: Key into the configs section
- `type`: Matcher type (currently only "glob")
- `pattern`: Glob pattern to match against module path
- `enabled`: Whether this matcher is active

### MatchContext

When matching, a `MatchContext` is created with:
- `module_path`: The fully qualified module path (e.g., "encoder.layers.5.mlp.fc1")
- `layer_number`: The layer number extracted from the path (if available)

### Loading Recipes

```bash
--quantization-recipe /path/to/recipe.yaml
```

Or programmatically:
```python
from megatron.core.quantization.utils import load_quantization_recipe, get_quant_config_or_none

recipe = load_quantization_recipe("recipe.yaml")
config = get_quant_config_or_none("encoder.layers.5.mlp.fc1", recipe)
```

## FP8 with MoE

FP8 quantization works with Mixture of Experts layers. Key considerations:

- Expert parallel parameters can be quantized with FP8
- The distributed optimizer handles FP8 for expert-parallel parameters with separate all-gather/reduce-scatter groups
- `fp8_param_gather` is compatible with expert parallelism
- Expert routing decisions remain in higher precision

## FP8 with Transformer Engine Integration

FP8 in Megatron-LM is built on Transformer Engine (TE). The integration points:

### fp8_autocast Context
```python
with transformer_engine.pytorch.fp8_autocast(
    enabled=True,
    fp8_recipe=recipe,
    fp8_group=amax_reduction_group,
):
    output = model(input)
```

### fp8_model_init Context
For initializing models with FP8 parameter storage:
```python
with transformer_engine.pytorch.fp8_model_init(
    enabled=True,
    recipe=recipe,
    preserve_high_precision_init_val=True,
):
    model = GPTModel(...)
```

### First/Last Layers in BF16

Some configurations keep the first and last layers in BF16 for stability:
```bash
--first-last-layers-bf16
--num-layers-at-start-in-bf16 1
--num-layers-at-end-in-bf16 1
```

This is not supported with delayed scaling (requires entering/exiting FP8 context per layer).

## FP8 Alignment Requirements

FP8 GEMMs require specific alignment:

| Recipe | Alignment Size |
|--------|---------------|
| Delayed / Tensorwise / Blockwise | 16 elements |
| MXFP8 | 32 elements |

When using FP8 inference, the `prepare_model_for_fp8_inference` function wraps TE linear layers to automatically pad/unpad sequences to meet alignment requirements.

## Memory Savings Summary

For a model with N parameters:

| Configuration | Param Memory | Optimizer Memory | Communication |
|---------------|-------------|------------------|---------------|
| BF16 baseline | 2N | 12N | 2N per all-gather |
| BF16 + DistOpt | 2N | 12N/D | 2N/D per all-gather |
| FP8 (params) + DistOpt | 1N | 12N/D | 1N/D per all-gather |
| FP8 (params+compute) + DistOpt | 1N | 12N/D | 1N/D per all-gather |

Where D = data-parallel world size. The FP8 parameter storage halves the communication volume compared to BF16.
