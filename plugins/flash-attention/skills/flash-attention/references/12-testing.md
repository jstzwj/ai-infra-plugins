# FlashAttention Testing Reference

This document provides comprehensive reference documentation for the testing infrastructure in FlashAttention. It covers test organization, execution, methodology, and common patterns.

---

## Table of Contents

1. [Overview](#overview)
2. [Test Organization](#test-organization)
3. [Running Tests](#running-tests)
4. [Fast Two-Pass Testing Methodology](#fast-two-pass-testing)
5. [Test Parametrization](#test-parametrization)
6. [Test Categories](#test-categories)
7. [Test Utilities](#test-utilities)
8. [Common Test Patterns](#common-test-patterns)
9. [CI/CD Integration](#cicd-integration)

---

## Overview

FlashAttention has a comprehensive test suite that validates:

- **Numerical correctness** against PyTorch reference implementations
- **Gradient accuracy** for all autograd functions
- **Memory efficiency** and absence of memory leaks
- **Edge cases** (variable-length sequences, padding, different dtypes)
- **Model integration** (loading pretrained weights, generation)
- **Performance regression** detection
- **Hardware compatibility** across GPU architectures (SM75 - SM100)

### Test File Summary

| Directory | Files | Purpose |
|-----------|-------|---------|
| `tests/` | Core attention tests | FlashAttention API validation |
| `tests/cute/` | CuTe DSL tests | FA4 kernel validation |
| `tests/models/` | Model tests | Model loading and inference |
| `tests/modules/` | Module tests | Parallel module validation |
| `tests/ops/` | Operation tests | Fused operation validation |
| `tests/layers/` | Layer tests | Rotary embedding tests |
| `tests/losses/` | Loss tests | Cross-entropy loss validation |

---

## Test Organization

### Directory Structure

```
tests/
    test_flash_attn.py                  # Main FlashAttention API tests
    test_flash_attn_ck.py               # CK (AMD) backend tests
    test_flash_attn_triton_amd.py       # Triton AMD backend tests
    test_rotary.py                      # Rotary embedding tests
    test_util.py                        # Utility tests

    cute/                               # CuTe DSL (FA4) tests
        conftest.py                     # Shared fixtures
        test_flash_attn.py              # Core FA4 attention tests
        test_flash_attn_fast.py         # Fast smoke tests
        test_flash_attn_varlen.py       # Variable-length tests
        test_flash_attn_combine.py      # Split-KV combine tests
        test_flash_attn_race_condition.py # Race condition tests
        test_clc_fuzz.py                # Fuzz testing
        test_mask_mod.py                # Mask modifier tests
        test_score_mod.py               # Score modifier tests
        test_score_mod_varlen.py        # Score modifier varlen tests
        test_block_sparsity.py          # Block sparsity tests
        test_cache_utils.py             # Cache utility tests
        test_utils.py                   # Test utility validation
        benchmark_block_sparsity.py     # Block sparsity benchmarks
        benchmark_mask_mod.py           # Mask modifier benchmarks
        mask_mod_definitions.py         # Mask modifier test definitions
        score_mod_definitions.py        # Score modifier test definitions

    models/                             # Model tests
        test_bert.py                    # BERT model tests
        test_gpt.py                     # GPT model tests
        test_gpt_generation_parallel.py # Parallel generation tests
        test_gpt_parallel.py            # Tensor parallel GPT tests
        test_gptj.py                    # GPT-J model tests
        test_gpt_neox.py                # GPT-NeoX model tests
        test_llama.py                   # LLaMA model tests
        test_opt.py                     # OPT model tests
        test_falcon.py                  # Falcon model tests
        test_vit.py                     # ViT model tests
        test_baichuan.py                # Baichuan model tests
        test_bigcode.py                 # BigCode model tests
        test_btlm.py                    # BTLM model tests

    modules/                            # Module tests
        test_mha_parallel.py            # Multi-head attention parallel
        test_mlp_parallel.py            # MLP parallel
        test_embedding_parallel.py      # Embedding parallel
        test_block_parallel.py          # Block parallel

    ops/                                # Operations tests
        test_fused_dense.py             # Fused dense tests
        test_fused_dense_parallel.py    # Parallel fused dense tests
        test_dropout_layer_norm.py       # Dropout+LayerNorm tests
        triton/
            test_layer_norm.py          # Triton layer norm tests

    layers/
        test_rotary.py                  # Rotary embedding layer tests

    losses/
        test_cross_entropy.py           # Cross-entropy loss tests
        test_cross_entropy_parallel.py  # Parallel cross-entropy tests
```

---

## Running Tests

### Basic Test Execution

```bash
# Run all core FlashAttention tests
pytest tests/test_flash_attn.py

# Run a specific test
pytest tests/test_flash_attn.py::test_flash_attn_output

# Run with verbose output
pytest tests/test_flash_attn.py -v

# Stop on first failure
pytest tests/test_flash_attn.py -x

# Run tests matching a pattern
pytest tests/test_flash_attn.py -k "causal"
```

### CuTe DSL (FA4) Tests

```bash
# Core attention tests
pytest tests/cute/test_flash_attn.py

# Single test
pytest tests/cute/test_flash_attn.py -k "test_flash_attn_output" -x

# Variable-length tests
pytest tests/cute/test_flash_attn_varlen.py

# Mask modifier tests
pytest tests/cute/test_mask_mod.py

# Score modifier tests
pytest tests/cute/test_score_mod.py

# Block sparsity tests
pytest tests/cute/test_block_sparsity.py
```

### Model Tests

```bash
# Test all models
pytest tests/models/

# Test specific model
pytest tests/models/test_gpt.py
pytest tests/models/test_llama.py
pytest tests/models/test_bert.py
```

### Operations Tests

```bash
# Fused dense operations
pytest tests/ops/test_fused_dense.py

# Layer norm operations
pytest tests/ops/test_dropout_layer_norm.py

# Triton layer norm
pytest tests/ops/triton/test_layer_norm.py
```

### GPU Selection

```bash
# Select specific GPU
CUDA_VISIBLE_DEVICES=0 pytest tests/test_flash_attn.py

# Find a free GPU
nvidia-smi  # Check GPU utilization
CUDA_VISIBLE_DEVICES=3 pytest tests/test_flash_attn.py
```

---

## Fast Two-Pass Testing Methodology

FlashAttention-4 (CuTe DSL) introduces a fast two-pass testing workflow that dramatically reduces total test time by separating kernel compilation from execution.

### The Problem

CuTe DSL kernels are JIT-compiled at runtime. Compilation time dominates total test time, especially when testing many parameter combinations. For example, compiling 200 kernel variants might take 30+ minutes, while executing them takes only seconds.

### The Solution

The two-pass workflow uses PyTorch's FakeTensorMode to compile kernels without GPU memory allocation, enabling massive parallelism during the compilation phase.

### Pass 1: Compile All Kernels (Parallel, No GPU Needed)

```bash
FLASH_ATTENTION_FAKE_TENSOR=1 \
FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
pytest -n 64 -x tests/cute/test_flash_attn.py
```

**Key environment variables:**
- `FLASH_ATTENTION_FAKE_TENSOR=1`: Uses `torch._subclasses.fake_tensor.FakeTensorMode` to compile kernels without allocating GPU memory or running them. The compilation happens entirely on CPU.
- `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1`: Enables persistent disk cache at `/tmp/${USER}/flash_attention_cute_dsl_cache/`. Compiled kernels are stored for reuse in Pass 2.
- `-n 64`: Uses pytest-xdist with 64 parallel workers. Since no GPU is needed, you can use as many workers as CPU cores allow.

**What happens:**
1. Each test case creates FakeTensor inputs (no GPU memory)
2. The kernel compilation pipeline runs normally
3. Compiled PTX/CUBIN is cached to disk
4. Tests "pass" by verifying compilation succeeds

### Pass 2: Execute Tests Using Cached Kernels

```bash
FLASH_ATTENTION_FAKE_TENSOR=0 \
FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
pytest -x tests/cute/test_flash_attn.py
```

**What happens:**
1. Tests load pre-compiled kernels from the disk cache
2. No compilation needed -- kernels are loaded and executed immediately
3. Full numerical validation runs on GPU

### Environment Variables Reference

| Variable | Values | Purpose |
|----------|--------|---------|
| `FLASH_ATTENTION_FAKE_TENSOR` | `0` (default), `1` | Enable FakeTensorMode for compilation-only |
| `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED` | `0` (default), `1` | Enable persistent disk cache |
| `CUTE_CUBIN_PATH` | Path string | Dump CUBIN/SASS for inspection |
| `CUTE_DSL_KEEP_PTX` | `1` | Keep intermediate PTX files |
| `FLASH_ATTENTION_DISABLE_BACKWARD` | `TRUE`/`FALSE` | Skip backward tests |

### Cache Location

Compiled kernels are cached at:
```
/tmp/${USER}/flash_attention_cute_dsl_cache/
```

Cache keys include:
- Data type (fp16, bf16)
- Head dimension (64, 96, 128)
- Causal/non-causal
- Mask/score modifier hashes
- Target architecture (SM90, SM100)
- Block sizes

---

## Test Parametrization

### Core Attention Test Parameters

Tests are parametrized over multiple dimensions:

```python
# Example parametrization from test_flash_attn.py
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("headdim", [64, 96, 128])
@pytest.mark.parametrize("seqlen", [128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_flash_attn_output(dtype, headdim, seqlen, causal, deterministic):
    ...
```

### CuTe DSL Test Parameters

FA4 tests are parametrized over:

| Parameter | Values | Notes |
|-----------|--------|-------|
| `dtype` | fp16, bf16 | Primary data types |
| `head_dim` | 64, 96, 128 | Standard head dimensions |
| `seq_len` | Various | 64 to 8192+ |
| `causal` | True, False | Causal attention |
| `num_heads_q` | Various | Query heads |
| `num_heads_kv` | Various | KV heads for GQA/MQA |
| `softmax_scale` | Various | Scale factor |

### Model Test Parameters

```python
@pytest.mark.parametrize("model_name", ["gpt2", "gpt2-medium", "gpt2-large"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_gpt_pretrained(model_name, dtype):
    ...
```

### Device Capability Filtering

Tests use device capability checks to skip unsupported configurations:

```python
is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)

@pytest.mark.skipif(not is_sm90, reason="Requires SM90")
def test_hopper_specific():
    ...
```

---

## Test Categories

### 1. Core Attention Tests (`tests/test_flash_attn.py`)

Tests for the FlashAttention-2 API.

**Key test functions:**
- `test_flash_attn_output` -- Validate output values match reference
- `test_flash_attn_grad` -- Validate gradients match reference
- `test_flash_attn_varlen` -- Variable-length sequence handling
- `test_flash_attn_kvpacked` -- Packed KV input format
- `test_flash_attn_qkvpacked` -- Packed QKV input format
- `test_flash_attn_alibi` -- ALiBi positional encoding
- `test_flash_attn_window_size` -- Sliding window attention
- `test_flash_attn_deterministic` -- Deterministic backward
- `test_flash_attn_with_kvcache` -- KV cache for generation

**Reference implementation:**
```python
def attention_ref(q, k, v, query_padding_mask=None, key_padding_mask=None,
                  attn_bias=None, dropout_p=0.0, causal=False, window_size=(-1, -1)):
    """Standard PyTorch attention for correctness checking."""
    q = q.float()
    k = k.float()
    v = v.float()
    # ... standard attention computation
```

### 2. CuTe DSL Tests (`tests/cute/`)

Tests for the FlashAttention-4 (CuTe DSL) implementation.

#### `test_flash_attn.py`

Core FA4 attention tests covering:
- Basic forward pass correctness
- Backward pass gradient correctness
- Various dtype/head_dim/causal combinations
- GQA (Grouped Query Attention)
- MQA (Multi-Query Attention)
- Softcap support

#### `test_flash_attn_varlen.py`

Variable-length sequence tests:
- Packed sequences with `cu_seqlens`
- Different sequence lengths within a batch
- Edge cases (single-token sequences, empty sequences)

#### `test_flash_attn_race_condition.py`

Tests for potential race conditions in concurrent kernel execution.

#### `test_clc_fuzz.py`

Fuzz testing with random inputs to find numerical edge cases.

#### `test_mask_mod.py`

Tests for user-defined mask modifiers:
- Causal masks
- Sliding window masks
- Custom boolean masks
- Block-sparse masks

#### `test_score_mod.py`

Tests for user-defined score modifiers:
- Softcap
- ALiBi
- Custom scaling functions

### 3. Model Tests (`tests/models/`)

Tests that validate model loading, forward pass, and weight conversion.

#### Common Pattern for Model Tests

```python
# tests/models/test_gpt.py
def test_gpt_pretrained():
    config = GPT2Config(...)
    model = GPTLMHeadModel(config)
    # Load from HuggingFace
    model = GPTLMHeadModel.from_pretrained("gpt2", config)
    # Verify forward pass
    input_ids = torch.randint(0, config.vocab_size, (2, 128))
    output = model(input_ids)
    assert output.logits.shape == (2, 128, config.vocab_size)
```

#### Model-Specific Tests

| Test File | Model | Key Tests |
|-----------|-------|-----------|
| `test_gpt.py` | GPT-2 | Pretrained loading, generation |
| `test_gpt_parallel.py` | GPT-2 TP | Tensor parallel forward/backward |
| `test_gpt_generation_parallel.py` | GPT-2 TP | Parallel generation |
| `test_llama.py` | LLaMA | Weight remapping, GQA |
| `test_bert.py` | BERT | MLM/NSP, padding handling |
| `test_gptj.py` | GPT-J | Parallel block architecture |
| `test_gpt_neox.py` | GPT-NeoX | Rotary embedding, parallel residual |
| `test_opt.py` | OPT | Post-norm, OPT-350m handling |
| `test_falcon.py` | Falcon | MQA, parallel attention |
| `test_vit.py` | ViT | Image classification, CLS pooling |
| `test_baichuan.py` | Baichuan | ALiBi/rotary variants |
| `test_bigcode.py` | BigCode | MQA weight tiling |
| `test_btlm.py` | BTLM | muP scaling, ALiBi |

### 4. Module Tests (`tests/modules/`)

Tests for tensor parallel module implementations.

- `test_mha_parallel.py` -- Parallel multi-head attention
- `test_mlp_parallel.py` -- Parallel MLP
- `test_embedding_parallel.py` -- Parallel embeddings
- `test_block_parallel.py` -- Parallel transformer blocks

### 5. Operations Tests (`tests/ops/`)

#### `test_fused_dense.py`

Tests for fused dense and MLP operations:
- FusedDense forward/backward
- FusedMLP forward/backward
- Different activation functions (gelu, relu, sqrelu)
- Gradient checkpointing levels (0, 1, 2)
- Return residual functionality

#### `test_fused_dense_parallel.py`

Tensor parallel tests for fused operations:
- ColumnParallelLinear
- RowParallelLinear
- ParallelFusedMLP
- Sequence parallel vs. non-sequence parallel

#### `test_dropout_layer_norm.py`

Tests for fused dropout + add + layer norm:
- DropoutAddLayerNorm
- DropoutAddRMSNorm
- Prenorm vs. postnorm
- Row/column scaling
- Parallel residual variant
- Subset variant

### 6. Loss Tests (`tests/losses/`)

#### `test_cross_entropy.py`

Cross-entropy loss tests:
- Basic correctness
- Label smoothing
- Logit scaling
- Z-loss regularization
- In-place backward
- Numerical stability

#### `test_cross_entropy_parallel.py`

Tensor parallel cross-entropy:
- Vocabulary sharding across ranks
- Correct loss aggregation
- Gradient correctness with sharding

---

## Test Utilities

### Attention Test Helpers (`tests/test_flash_attn.py`)

#### `generate_random_padding_mask`

```python
def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    """Generate random padding masks for testing.
    Modes: 'full' (no padding), 'random', 'third' (1/3 to max)
    """
```

#### `generate_qkv`

```python
def generate_qkv(q, k, v, query_padding_mask=None, key_padding_mask=None,
                 kvpacked=False, qkvpacked=False):
    """Generate QKV tensors in various formats for testing.
    Returns padded and unpadded versions with appropriate indices.
    """
```

#### `attn_bias_from_alibi_slopes`

```python
def attn_bias_from_alibi_slopes(slopes, seqlen_q, seqlen_k, ...):
    """Generate attention bias from ALiBi slopes for reference computation."""
```

### CuTe Test Fixtures (`tests/cute/conftest.py`)

Shared pytest fixtures for CuTe DSL tests:
- GPU device selection
- Dtype fixtures
- Common parameter sets

### Mask/Score Definitions

**File:** `tests/cute/mask_mod_definitions.py`

Library of mask modifier definitions used across tests:
- Causal mask
- Local/sliding window mask
- Block diagonal mask
- Prefix mask

**File:** `tests/cute/score_mod_definitions.py`

Library of score modifier definitions:
- Softcap
- ALiBi
- Temperature scaling

---

## Common Test Patterns

### Numerical Correctness Testing

The most common test pattern compares FlashAttention output against a reference PyTorch implementation:

```python
def test_flash_attn_output():
    # Setup
    q = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=dtype,
                     requires_grad=True)
    k = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=dtype)

    # FlashAttention
    out_fa = flash_attn_func(q, k, v, causal=causal)

    # Reference
    out_ref = attention_ref(q, k, v, causal=causal)

    # Compare
    assert torch.allclose(out_fa, out_ref, atol=1e-2, rtol=1e-2)
```

### Gradient Correctness Testing

```python
def test_flash_attn_grad():
    q = torch.randn(..., requires_grad=True)
    k = torch.randn(...)
    v = torch.randn(...)

    out = flash_attn_func(q, k, v, causal=True)
    loss = out.sum()
    loss.backward()

    # Compare with numerical gradient or reference gradient
    grad_ref = compute_reference_gradient(q, k, v)
    assert torch.allclose(q.grad, grad_ref, atol=1e-2, rtol=1e-2)
```

### Variable-Length Testing

```python
def test_flash_attn_varlen():
    # Create packed sequences
    cu_seqlens = torch.tensor([0, 128, 256, 384, 512], device="cuda", dtype=torch.int32)
    total_len = cu_seqlens[-1]
    q = torch.randn(total_len, nheads, headdim, device="cuda", dtype=dtype)

    out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens,
                                  max_seqlen=max_seqlen)
```

### Model Weight Loading Testing

```python
def test_model_from_pretrained():
    """Verify that pretrained weights are correctly loaded and remapped."""
    # Load with HuggingFace
    hf_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load with FlashAttention
    fa_model = GPTLMHeadModel.from_pretrained(model_name, config)

    # Compare outputs
    input_ids = torch.randint(0, vocab_size, (1, 64))
    with torch.no_grad():
        hf_out = hf_model(input_ids).logits
        fa_out = fa_model(input_ids).logits

    assert torch.allclose(hf_out, fa_out, atol=1e-2, rtol=1e-2)
```

### Tensor Parallel Testing

```python
def test_tensor_parallel():
    """Test that tensor parallel produces same results as single GPU."""
    # Full model
    full_model = create_model(config)
    full_out = full_model(input_ids)

    # Sharded model (simulated)
    state_dicts = [shard_state_dict_tp(full_state_dict, config, world_size, rank)
                   for rank in range(world_size)]
    for rank in range(world_size):
        model = create_model(config, process_group=pg)
        model.load_state_dict(state_dicts[rank])
        # Run forward, compare with full model
```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# Example CI configuration
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run core tests
        run: pytest tests/test_flash_attn.py -x --tb=short
      - name: Run model tests
        run: pytest tests/models/ -x --tb=short
      - name: Run ops tests
        run: pytest tests/ops/ -x --tb=short
```

### Test Selection by GPU

```python
# Skip tests based on GPU capability
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.skipif(torch.cuda.get_device_capability() < (8, 0), reason="Requires SM80+")
def test_sm80_only():
    ...
```

### Parallel Test Execution

For large test suites, use pytest-xdist with GPU affinity:

```bash
# Run tests in parallel across GPUs
CUDA_VISIBLE_DEVICES=0,1 pytest -n 2 tests/ops/ --dist=loadfile

# For CuTe DSL compilation phase (CPU only, high parallelism)
FLASH_ATTENTION_FAKE_TENSOR=1 FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
    pytest -n 256 tests/cute/test_flash_attn.py
```

### Debugging Failed Tests

1. **Verbose output**: `pytest -vv`
2. **Print statements**: `pytest -s` (show stdout)
3. **Tracebacks**: `pytest --tb=long`
4. **Single test**: Run only the failing test with `-k`
5. **GPU debugging**: Use `CUTE_DSL_KEEP_PTX=1` and `CUTE_DSL_LINEINFO=1` for kernel inspection
6. **Race conditions**: Use `compute-sanitizer --tool=racecheck` (beware false positives with TMA)
