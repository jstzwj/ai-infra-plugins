# bitsandbytes: Source File Index

This document provides a complete index of all source files in the bitsandbytes repository, organized by module. Each entry includes the file path and a brief description of its contents.

---

## Root Files

| File | Description |
|---|---|
| `bitsandbytes/__init__.py` | Package initialization. Loads backends (CUDA, XPU, MPS, HPU), exports `features`, `supported_torch_devices`, imports autograd functions (`MatmulLtState`, `matmul`, `matmul_4bit`). Defines `__version__ = "0.50.0.dev0"`. |
| `bitsandbytes/functional.py` | Core quantization/dequantization primitives: `QuantState` class, `quantize_blockwise`, `dequantize_blockwise`, `quantize_4bit`, `dequantize_4bit`, `quantize_fp4`, `quantize_nf4`, `gemv_4bit`, `int8_linear_matmul`, `int8_mm_dequant`, `int8_vectorwise_quant`, `int8_vectorwise_dequant`, `int8_double_quant`, `optimizer_update_32bit`, `optimizer_update_8bit_blockwise`, `igemm`, `batched_igemm`. Quantization map generators: `create_normal_map`, `create_fp8_map`, `create_dynamic_map`, `create_linear_map`. Helper functions: `get_ptr`, `is_on_gpu`, `get_paged`, `prefetch_tensor`, `_convert_weight_packed_for_cpu`, `has_avx512bf16`. Singletons: `GlobalPageManager`, `CUBLAS_Context`. |
| `bitsandbytes/_ops.py` | PyTorch custom op definitions (`torch.library.define`) and fake implementations (`register_fake`) for all bnb operations. Enables `torch.compile` compatibility. Defines 15 custom ops including `int8_mixed_scaled_mm`, `int8_scaled_mm`, `int8_linear_matmul`, `int8_vectorwise_quant`, `int8_vectorwise_dequant`, `int8_mm_dequant`, `int8_double_quant`, `dequantize_4bit`, `quantize_4bit`, `dequantize_blockwise`, `quantize_blockwise`, `gemv_4bit`, `optimizer_update_32bit`, `optimizer_update_8bit_blockwise`. Handles PyTorch version compatibility (2.3 vs 2.4+). |
| `bitsandbytes/utils.py` | Utility functions: `replace_linear` (recursively replaces `nn.Linear` with a custom class), `pack_dict_to_tensor` / `unpack_tensor_to_dict` (JSON serialization of quant_state for safetensors), `OutlierTracer` (singleton for detecting outlier dimensions across layers), `find_outlier_dims`, `execute_and_return`, `sync_gpu`. Constants: `LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING`, `INVERSE_LINEAR_8BIT_WEIGHTS_FORMAT_MAPPING`. |
| `bitsandbytes/cextension.py` | C library loading and error handling. Classes: `BNBNativeLibrary`, `CudaBNBNativeLibrary`, `XpuBNBNativeLibrary`, `ErrorHandlerMockBNBNativeLibrary`. Functions: `get_native_library`, `get_cuda_bnb_library_path`, `get_available_cuda_binary_versions`, `parse_cuda_version`. Handles `BNB_CUDA_VERSION` and `BNB_ROCM_VERSION` environment variable overrides. Falls back to mock library with detailed error messages on failure. |
| `bitsandbytes/cuda_specs.py` | CUDA specification detection. `CUDASpecs` frozen dataclass with `highest_compute_capability`, `cuda_version_string`, `cuda_version_tuple`, and `has_imma` property. Functions: `get_compute_capabilities`, `get_cuda_version_tuple`, `get_cuda_version_string`, `get_cuda_specs`, `get_rocm_gpu_arch`. Cached results via `@lru_cache`. |
| `bitsandbytes/consts.py` | Constants: `DYNAMIC_LIBRARY_SUFFIX` (platform-specific `.so`/`.dylib`/`.dll`), `PACKAGE_DIR` (path to package root), `PACKAGE_GITHUB_URL`, `NONPYTORCH_DOC_URL`. |
| `bitsandbytes/__main__.py` | CLI entry point. Runs `python -m bitsandbytes` to execute diagnostics. Calls `bitsandbytes.diagnostics.main.main()`. |

---

## Neural Network Module (`bitsandbytes/nn/`)

| File | Description |
|---|---|
| `nn/__init__.py` | Module exports: `StableEmbedding`, `Embedding`, `Embedding4bit`, `Embedding8bit`, `EmbeddingFP4`, `EmbeddingNF4`, `Int8Params`, `Params4bit`, `Linear4bit`, `Linear8bitLt`, `LinearFP4`, `LinearNF4`, `OutlierAwareLinear`. |
| `nn/modules.py` | Core neural network modules. **Linear layers**: `Linear4bit` (base 4-bit class with `Params4bit` weight, CPU AVX512-BF16 optimization, FSDP state recovery), `LinearFP4` (FP4 variant), `LinearNF4` (NF4 variant), `Linear8bitLt` (8-bit with `Int8Params`, `MatmulLtState`, mixed-precision decomposition). **Parameters**: `Params4bit` (4-bit quantized parameter with `@property` FSDP proxies for `absmax`, `code`, `quant_map`, `offset`, `state2`, `nested_*`), `Int8Params` (8-bit quantized parameter with `CB`, `SCB`). **Embeddings**: `StableEmbedding` (32-bit optimizer states + LayerNorm), `Embedding` (base with 32-bit optimizer override), `Embedding8bit` (8-bit quantized), `Embedding4bit` (4-bit quantized with partial dequantize optimization), `EmbeddingFP4`, `EmbeddingNF4`. **Utilities**: `fix_4bit_weight_quant_state_from_module` (recovers lost quant_state), `maybe_rearrange_weight` (state_dict hook), `OutlierAwareLinear`. |
| `nn/parametrize.py` | PyTorch parametrization support for 4-bit quantization. `Bnb4bitParametrization` (dequantization module), `replace_parameter_4bit` (quantizes a parameter and registers parametrization), `replace_parameter_4bit_prequantized` (for pre-quantized data). Manages forward hooks for parametrization caching and state_dict post-hooks for quantization state serialization. |

---

## Autograd Functions (`bitsandbytes/autograd/`)

| File | Description |
|---|---|
| `autograd/__init__.py` | Exports (empty, imports are done from top-level `bitsandbytes`). |
| `autograd/_functions.py` | Autograd function implementations. **`MatmulLtState`** dataclass: manages 8-bit matmul state with `CB`, `SB`, `SCB`, `SBt`, `CBt`, `subB`, `outlier_pool`, deprecated fields (`CxB`, `CxBt`, `formatB`, `_tile_indices`). **`MatMul8bitLt`**: autograd function for 8-bit matmul with mixed-precision decomposition (handles forward with int8_scaled_mm/int8_mixed_scaled_mm, backward with gradient computation through dequantized weights). **`MatMul8bitFp`**: simpler 8-bit autograd for CPU/XPU that dequantizes then uses standard matmul. **`MatMul4Bit`**: autograd function for 4-bit matmul (dequantize then standard linear). **`matmul`**: dispatches to `MatMul8bitLt` or `MatMul8bitFp` based on device and training mode. **`matmul_4bit`**: dispatches to `gemv_4bit` (fast path for single-batch inference) or `MatMul4Bit.apply`. **`GlobalOutlierPooler`**: singleton for pooling outlier dimensions across layers. |

---

## Optimizers (`bitsandbytes/optim/`)

| File | Description |
|---|---|
| `optim/__init__.py` | Exports all optimizer classes: `Adagrad/Adagrad8bit/Adagrad32bit`, `Adam/Adam8bit/Adam32bit/PagedAdam/PagedAdam8bit/PagedAdam32bit`, `AdamW/AdamW8bit/AdamW32bit/PagedAdamW/PagedAdamW8bit/PagedAdamW32bit`, `AdEMAMix/AdEMAMix8bit/AdEMAMix32bit/PagedAdEMAMix/PagedAdEMAMix8bit/PagedAdEMAMix32bit`, `LAMB/LAMB8bit/LAMB32bit`, `LARS/LARS8bit/LARS32bit/PytorchLARS`, `Lion/Lion8bit/Lion32bit/PagedLion/PagedLion8bit/PagedLion32bit`, `RMSprop/RMSprop8bit/RMSprop32bit`, `SGD/SGD8bit/SGD32bit`, `GlobalOptimManager`. |
| `optim/optimizer.py` | Base optimizer classes. **`GlobalOptimManager`**: singleton for per-parameter optimizer config overrides (32-bit states for embeddings). **`Optimizer8bit`**: base class with FSDP-compatible `state_dict`/`load_state_dict`, paged memory support, `get_state_buffer` (allocates paged or standard buffers). **`Optimizer2State`**: base for 2-state optimizers (Adam, AdamW, etc.) with `init_state` and `update_step` that dispatch to `optimizer_update_32bit` or `optimizer_update_8bit_blockwise`. **`Optimizer1State`**: base for 1-state optimizers (SGD, etc.). **`MockArgs`**: simple args container. |
| `optim/adam.py` | Adam optimizer variants: `Adam` (standard PyTorch Adam), `Adam8bit` (8-bit optimizer states), `Adam32bit` (explicit 32-bit), `PagedAdam` (paged memory), `PagedAdam8bit` (paged + 8-bit), `PagedAdam32bit` (paged + 32-bit). |
| `optim/adamw.py` | AdamW optimizer variants: `AdamW`, `AdamW8bit`, `AdamW32bit`, `PagedAdamW`, `PagedAdamW8bit`, `PagedAdamW32bit`. |
| `optim/sgd.py` | SGD optimizer variants: `SGD`, `SGD8bit`, `SGD32bit`. |
| `optim/lion.py` | Lion optimizer variants: `Lion`, `Lion8bit`, `Lion32bit`, `PagedLion`, `PagedLion8bit`, `PagedLion32bit`. |
| `optim/lamb.py` | LAMB optimizer variants: `LAMB`, `LAMB8bit`, `LAMB32bit`. |
| `optim/lars.py` | LARS optimizer variants: `LARS`, `LARS8bit`, `LARS32bit`, `PytorchLARS`. |
| `optim/adagrad.py` | Adagrad optimizer variants: `Adagrad`, `Adagrad8bit`, `Adagrad32bit`. |
| `optim/rmsprop.py` | RMSprop optimizer variants: `RMSprop`, `RMSprop8bit`, `RMSprop32bit`. |
| `optim/ademamix.py` | AdEMAMix optimizer variants (3-state): `AdEMAMix`, `AdEMAMix8bit`, `AdEMAMix32bit`, `PagedAdEMAMix`, `PagedAdEMAMix8bit`, `PagedAdEMAMix32bit`. |

---

## Backends (`bitsandbytes/backends/`)

| File | Description |
|---|---|
| `backends/__init__.py` | Backend module exports (empty). |
| `backends/utils.py` | Shared backend utility functions. |
| `backends/default/__init__.py` | Default backend initialization. |
| `backends/default/ops.py` | Fallback implementations for all ops. Used when no hardware-specific backend is available. Provides PyTorch-native implementations of quantize, dequantize, matmul, and optimizer operations. |
| `backends/cuda/__init__.py` | CUDA backend initialization. |
| `backends/cuda/ops.py` | CUDA-specific implementations. Registers kernels for `bitsandbytes::*` custom ops on CUDA dispatch key. Uses the loaded native library (`lib`) for C++/CUDA kernel calls. |
| `backends/cpu/__init__.py` | CPU backend initialization. |
| `backends/cpu/ops.py` | CPU-specific implementations. Registers kernels for CPU dispatch key. Uses AVX512-BF16 instructions when available for optimized 4-bit inference. |
| `backends/triton/__init__.py` | Triton backend initialization. |
| `backends/triton/ops.py` | Triton-specific implementations. Registers kernels using Triton JIT-compiled kernels for GPU operations. |
| `backends/triton/kernels_4bit.py` | Triton kernel for 4-bit dequantization. Implements blockwise dequantization with configurable blocksize and quant_type (nf4/fp4). |
| `backends/triton/kernels_8bit_quant.py` | Triton kernel for 8-bit quantization. Implements vectorwise quantization with optional outlier detection. |
| `backends/triton/kernels_optim.py` | Triton kernels for optimizer operations. Implements 8-bit blockwise optimizer updates. |
| `backends/mps/__init__.py` | Metal Performance Shaders backend initialization. |
| `backends/mps/ops.py` | Apple Silicon MPS implementations. Registers kernels for MPS dispatch key using Metal shaders. |
| `backends/xpu/__init__.py` | Intel XPU backend initialization. |
| `backends/xpu/ops.py` | Intel XPU implementations. Registers kernels for XPU dispatch key using SYCL/DPC++ operations. |
| `backends/hpu/__init__.py` | Intel Gaudi HPU backend initialization. |
| `backends/hpu/ops.py` | Intel Gaudi HPU implementations. Registers kernels for HPU dispatch key. |

---

## Diagnostics (`bitsandbytes/diagnostics/`)

| File | Description |
|---|---|
| `diagnostics/__init__.py` | Diagnostics module initialization. |
| `diagnostics/main.py` | Main diagnostics entry point. `show_environment()` prints platform, Python, PyTorch, and related package versions. `sanity_check()` runs a quick Adam optimizer step to verify functionality. `main()` orchestrates the full diagnostics output (called by `python -m bitsandbytes`). |
| `diagnostics/cuda.py` | CUDA diagnostics. `print_diagnostics()` prints CUDA/ROCm binary path and compute capability. `find_cudart_libraries()` searches for CUDA runtime libraries in environment paths. `find_cuda_libraries_in_path_list()` scans directory paths for runtime library files. Handles both CUDA and ROCm diagnostic paths. |
| `diagnostics/utils.py` | Utility functions: `print_header()` (formatted header with separator), `print_dedented()` (textwrap.dedent wrapper). |

---

## CUDA/C++ Source (`csrc/`)

| File | Description |
|---|---|
| `pythonInterface.cpp` | Pybind11/C interface. Exposes C++/CUDA functions to Python via ctypes. Defines the `get_context()` and `cget_managed_ptr()` functions, plus all optimizer, quantization, and matmul operation bindings. |
| `ops.cu` | CUDA operation implementations. Contains the CUDA kernels for int8 matmul, int8 vectorwise quant/dequant, int8 double quant, 4-bit quant/dequant, blockwise quant/dequant, and optimizer updates. |
| `ops.cuh` | CUDA operation headers. Declares the kernel function signatures for ops.cu. |
| `kernels.cu` | Lower-level CUDA kernels. Contains element-wise operations (fill, mul), paged memory operations (prefetch, managed pointer allocation), and utility kernels. |
| `kernels.cuh` | Kernel headers. Declares the kernel function signatures for kernels.cu. |
| `common.h` | Common C++ definitions. Shared type definitions, constants, and utility macros used across C++ source files. |
| `common.cuh` | Common CUDA definitions. Shared CUDA-specific type definitions, device functions, and macros. |
| `compat.cuh` | Compatibility headers. Provides compatibility macros and functions for different CUDA versions and compute capabilities. |
| `compat_device.cuh` | Device compatibility headers. Handles differences between CUDA, ROCm, and other GPU platforms at the device level. |
| `cpu_ops.cpp` | CPU operation implementations. Contains C++ implementations of quantization, dequantization, and optimizer operations for CPU execution. Includes AVX512-BF16 optimized paths for 4-bit inference. |
| `cpu_ops.h` | CPU operation headers. Declares the CPU operation function signatures. |
| `xpu_ops.cpp` | Intel XPU operation implementations. Contains SYCL/DPC++ implementations of quantization and optimizer operations for Intel GPUs. |
| `xpu_ops.h` | Intel XPU operation headers. Declares the XPU operation function signatures. |
| `xpu_kernels.cpp` | Intel XPU kernel implementations. Lower-level SYCL kernels for XPU operations. |
| `xpu_kernels.h` | Intel XPU kernel headers. Declares the XPU kernel function signatures. |
| `mps_kernels.metal` | Apple Metal shader implementations. Contains Metal Shading Language kernels for Apple Silicon GPU operations. |
| `mps_ops.mm` | Apple Metal operation implementations. Objective-C++ bridge between Python and Metal shaders. |

---

## Tests (`tests/`)

| File | Description |
|---|---|
| `__init__.py` | Test package initialization. |
| `conftest.py` | Pytest configuration. Defines shared fixtures for device detection, dtype parameters, and model setup. |
| `helpers.py` | Test helper functions. Shared utilities for creating test tensors, comparing results, and setting up test models. |
| `test_functional.py` | Tests for `functional.py`: blockwise quantize/dequantize, 4-bit quantize/dequantize (NF4, FP4), int8 vectorwise quant/dequant, int8 double quant, int8 matmul, gemv_4bit, optimizer updates, igemm, batched igemm, create_normal_map, create_fp8_map, create_dynamic_map, QuantState serialization. |
| `test_linear4bit.py` | Tests for `Linear4bit`, `LinearFP4`, `LinearNF4`: forward pass correctness, backward pass, device movement (CPU/GPU), state_dict save/load, double quantization, different compute dtypes, FSDP compatibility, CPU inference with AVX512-BF16. |
| `test_linear8bitlt.py` | Tests for `Linear8bitLt`: forward pass with/without outliers, backward pass, device movement, state_dict save/load, mixed-precision decomposition with threshold, has_fp16_weights modes, weight format handling. |
| `test_modules.py` | Tests for nn modules: `StableEmbedding`, `Embedding`, `Embedding8bit`, `Embedding4bit`, `EmbeddingFP4`, `EmbeddingNF4`, `Params4bit` (serialization, device movement, `from_prequantized`), `Int8Params` (quantization, device movement), `OutlierAwareLinear`. |
| `test_autograd.py` | Tests for autograd functions: `MatMul8bitLt` forward/backward, `MatMul4Bit` forward/backward, `MatMul8bitFp` forward/backward, `matmul` dispatch, `matmul_4bit` dispatch and fast-path selection, `MatmulLtState` deprecated field warnings. |
| `test_ops.py` | Tests for PyTorch custom ops: fake implementation correctness, shape inference, dtype validation, device placement for all 15 registered custom ops. |
| `test_optim.py` | Tests for all optimizer variants: Adam, AdamW, SGD, Lion, LAMB, LARS, Adagrad, RMSprop, AdEMAMix and their 8-bit, 32-bit, and paged variants. Tests convergence, gradient handling, state dict serialization, `GlobalOptimManager` overrides, `min_8bit_size` behavior. |
| `test_parametrize.py` | Tests for `Bnb4bitParametrization`, `replace_parameter_4bit`, `replace_parameter_4bit_prequantized`: parametrization registration, dequantization in forward, state_dict save/load with quantization state, forward hooks for caching. |
| `test_generation.py` | End-to-end text generation tests: load a quantized model, generate text, verify output quality and shape. Tests both 4-bit and 8-bit pipelines. |
| `test_cuda_setup_evaluator.py` | Tests for CUDA setup and device detection: `get_cuda_specs`, `get_compute_capabilities`, `get_cuda_version_tuple`, library path detection. |
| `fsdp_state_dict_save.py` | FSDP state dict save/load test script. Verifies that quantized model state dicts can be saved and loaded correctly with PyTorch FSDP. |

---

## Examples (`examples/`)

| File | Description |
|---|---|
| `int8_inference_huggingface.py` | 8-bit inference example using HuggingFace Transformers. Demonstrates loading a model with `BitsAndBytesConfig(load_in_8bit=True)` and running inference. |
| `compile_inference.py` | `torch.compile` inference example. Shows how to use `torch.compile` with bitsandbytes quantized models, leveraging the custom op fake implementations. |
| `cpu/cpu_training.py` | CPU training example. Demonstrates using 8-bit optimizers for training on CPU with AVX512-BF16 optimizations. |
| `xpu/paged_xpu_training.py` | Paged XPU training example. Demonstrates using paged optimizers on Intel XPU devices for memory-efficient training. |
| `xpu/benchmark_paged_memory.py` | Paged memory benchmark for XPU. Compares memory usage and performance of paged vs non-paged optimizers on Intel GPUs. |

---

## Benchmarking (`benchmarking/`)

| File | Description |
|---|---|
| `inference_benchmark.py` | Inference speed benchmark. Measures latency and throughput for quantized model inference across different batch sizes and sequence lengths. |
| `matmul_benchmark.py` | Matmul performance benchmark. Compares int8, 4-bit, and fp16 matmul performance across different matrix sizes. |
| `optimizer_benchmark.py` | Optimizer performance benchmark. Compares 8-bit, 32-bit, and paged optimizer variants for speed and memory usage. |
| `int8/training_benchmark.py` | INT8 training benchmark. Measures training throughput with 8-bit quantized layers. |
| `int8/int8_benchmark.py` | INT8 matmul benchmark. Detailed performance measurements for int8 matrix multiplication kernels. |
| `xpu/inference_benchmark.py` | XPU inference benchmark. Measures inference performance on Intel XPU devices. |

---

## Documentation Source (`docs/source/`)

| File | Description |
|---|---|
| `index.mdx` | Documentation index page. |
| `quickstart.mdx` | Quick start guide. Introduction to bitsandbytes features with basic examples. |
| `installation.mdx` | Installation guide. Covers pip install, compiling from source, CUDA version selection, platform-specific instructions. |
| `optimizers.mdx` | Optimizer overview. Introduction to 8-bit optimizers and their benefits. |
| `integrations.mdx` | Integration guides. How to use bitsandbytes with Transformers, PEFT, Diffusers. |
| `contributing.mdx` | Contribution guide. Development setup, pre-commit hooks, PR workflow. |
| `errors.mdx` | Error documentation. Common errors and their solutions. |
| `faqs.mdx` | Frequently asked questions. |
| `fsdp_qlora.md` | FSDP + QLoRA guide. How to use Fully Sharded Data Parallel with 4-bit quantized models. |
| `_toctree.yml` | Documentation table of contents configuration. |
| `reference/functional.mdx` | Functional API reference. Documents all public functions in `functional.py`. |
| `reference/nn/linear4bit.mdx` | `Linear4bit` API reference. |
| `reference/nn/linear8bit.mdx` | `Linear8bitLt` API reference. |
| `reference/nn/embeddings.mdx` | Embedding module API reference. |
| `reference/optim/optim_overview.mdx` | Optimizer overview reference. |
| `reference/optim/adam.mdx` | Adam optimizer reference. |
| `reference/optim/adamw.mdx` | AdamW optimizer reference. |
| `reference/optim/sgd.mdx` | SGD optimizer reference. |
| `reference/optim/lion.mdx` | Lion optimizer reference. |
| `reference/optim/lamb.mdx` | LAMB optimizer reference. |
| `reference/optim/lars.mdx` | LARS optimizer reference. |
| `reference/optim/adagrad.mdx` | Adagrad optimizer reference. |
| `reference/optim/rmsprop.mdx` | RMSprop optimizer reference. |
| `reference/optim/ademamix.mdx` | AdEMAMix optimizer reference. |
| `explanations/optimizers.mdx` | In-depth explanation of how 8-bit optimizers work. Blockwise quantization, dynamic data types, paged memory. |
| `explanations/resources.mdx` | Resource usage guide. Memory requirements for different quantization configurations. |

---

## Agent Guides (`agents/`)

| File | Description |
|---|---|
| `architecture_guide.md` | Codebase architecture overview. Module organization, data flow, key design patterns, threading model. |
| `api_surface.md` | Public API catalog. Documents all public classes, functions, and attributes that downstream projects depend on. Used for breaking-change detection during PR reviews. |
| `code_standards.md` | Code quality standards. Naming conventions, documentation requirements, testing expectations, review criteria. |
| `pr_review_guide.md` | PR review workflow. Step-by-step process for reviewing pull requests: classification, checklist, verdict format, posting instructions. References all other agent guides. |
| `security_guide.md` | Security trust model and checklist. Especially relevant for external contributor PRs. Covers native library loading, input validation, memory safety. |
| `testing_guide.md` | Testing practices and known issues. How to run specific tests, common test patterns, flaky tests, platform-specific considerations. |
| `linting_guide.md` | Linting configuration and troubleshooting. Full pre-commit hook details, ruff rules, clang-format setup. |
| `dispatch_guide.md` | Issue dispatch workflow. How to triage open GitHub issues, generate prompt files, and launch parallel worker agents. |
| `worktree_guide.md` | Git worktree management. Naming conventions, creation/removal, parallel session handling. |
| `downstream_integrations.md` | Downstream integration catalog. Detailed documentation of how Transformers, PEFT, Accelerate, TGI, and vLLM depend on specific bnb APIs. Consolidated breaking-change surface tables. |
| `issue_maintenance_guide.md` | Issue maintenance guide. How to identify and close stale, duplicate, or resolved issues. |
| `issue_triage_workflow.md` | Issue triage process. Labeling, prioritization, and assignment guidelines. |
| `issue_patterns.md` | Common closeable issue patterns. Old CUDA setup issues, Windows pre-support issues, third-party app issues, etc. |
| `github_tools_guide.md` | GitHub CLI tools reference. `gh` commands for issue management, PR review, and CI interaction. |
| `fetch_issues.py` | GitHub issue fetching script. Automated tool for pulling issue data from the repository. |
| `query_issues.py` | GitHub issue query script. Search and filter issues based on various criteria. |

---

## Build and Configuration Files

| File | Description |
|---|---|
| `pyproject.toml` | Project metadata, dependencies, build configuration, tool settings (pytest, ruff, mypy, coverage, scikit-build). |
| `CMakeLists.txt` | CMake build configuration for CUDA/C++ native libraries. |
| `setup.py` | Legacy setup script (kept for backward compatibility). |
| `install_cuda.py` | CUDA installation helper script. |
| `install_cuda.sh` | CUDA installation shell script. |
| `check_bnb_install.py` | Installation verification script. Runs a quick sanity check to verify bitsandbytes is properly installed. |
| `_typos.toml` | Typos spell-checker configuration. |
| `MANIFEST.in` | Package manifest for source distribution. |
| `CLAUDE.md` | Agent instructions for Claude Code. Mandatory worktree usage, testing guidelines, agent dispatch instructions, PR review workflow. |
| `.github/scripts/set_platform_tag.py` | CI script to set platform tags on build wheels. |
| `.github/scripts/auditwheel_show.py` | CI script to audit wheel dependencies. |

---

## Additional Repository Files

| File | Description |
|---|---|
| `README.md` | Project readme with overview, installation, and usage examples. |
| `CHANGELOG.md` | Version history and release notes. |
| `CONTRIBUTING.md` | Contribution guidelines (brief version). |
| `LICENSE` | MIT License. |
| `CODE_OF_CONDUCT.md` | Code of conduct. |
| `SECURITY.md` | Security policy. |
| `NOTICE.md` | Legal notices. |
| `COMPILE_H100_L40.md` | Compilation instructions for H100 and L40 GPUs. |
