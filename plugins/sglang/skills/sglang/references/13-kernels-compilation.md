# SGLang Kernels and Compilation Reference

This document provides a comprehensive reference for the kernel library, CUDA graph support, torch.compile integration, and compilation pipeline in SGLang. These components are critical for achieving high performance in LLM inference.

## Table of Contents

- [Overview](#overview)
- [sgl-kernel Package](#sgl-kernel-package)
- [CUDA Graph Support](#cuda-graph-support)
- [Standard CUDA Graphs](#standard-cuda-graphs)
- [Breakable CUDA Graph](#breakable-cuda-graph)
- [Piecewise CUDA Graph](#piecewise-cuda-graph)
- [torch.compile Integration](#torchcompile-integration)
- [Piecewise CUDA Graph Compiler](#piecewise-cuda-graph-compiler)
- [Compilation Pipeline](#compilation-pipeline)
- [DeepGEMM Kernels](#deepgemm-kernels)
- [FlashInfer Integration](#flashinfer-integration)
- [Triton Kernels](#triton-kernels)
- [CUTLASS Integration](#cutlass-integration)
- [JIT Kernel Compilation](#jit-kernel-compilation)
- [Performance Optimization Tips](#performance-optimization-tips)
- [Source Code Structure](#source-code-structure)

---

## Overview

SGLang's high performance depends on a carefully engineered stack of custom CUDA kernels, graph capture mechanisms, and compilation pipelines. The main components are:

1. **sgl-kernel**: A standalone kernel library providing optimized CUDA compute primitives for LLM inference
2. **CUDA Graphs**: Multiple graph capture strategies (standard, breakable, piecewise) to eliminate kernel launch overhead
3. **torch.compile**: Integration with PyTorch's compilation framework for operator fusion and autotuning
4. **External kernel libraries**: DeepGEMM, FlashInfer, CUTLASS, and Triton for specialized operations

---

## sgl-kernel Package

`sglang-kernel` (imported as `sgl_kernel`) is the dedicated kernel library for LLM inference. It provides optimized compute primitives through custom CUDA kernel operations.

### Installation

```bash
# Latest version (requires torch == 2.11.0)
pip3 install sglang-kernel --upgrade
```

### Building from Source

Requirements:
- CMake >= 3.31
- Python >= 3.10
- scikit-build-core
- ninja (optional)

```bash
# From sgl-kernel source tree
make build

# Limit build parallelism
make build MAX_JOBS=2

# Additionally limit NVCC internal threads
make build MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
```

### Kernel Categories

The kernel library provides operations across multiple categories:

| Category | Directory | Description |
|----------|-----------|-------------|
| AllReduce | `csrc/allreduce/` | Custom all-reduce and MSCCL++ all-reduce |
| Attention | `csrc/attention/` | CUTLASS MLA kernel, merge attention states, vertical/slash index |
| Elementwise | `csrc/elementwise/` | Activation, concat, copy, fused add RMS norm, positional encoding, top-k |
| Expert Specialization | `csrc/expert_specialization/` | FP8 blockwise, SM100 MXFP8 block-scaled operations |
| GEMM | `csrc/gemm/` | AWQ, BMM FP8, DSV3 fused/router GEMM, FP8 blockwise, INT8, per-token quant, GPTQ, QServe |
| Grammar | `csrc/grammar/` | Token bitmask application |
| KV Cache I/O | `csrc/kvcacheio/` | KV cache transfer operations |
| Mamba | `csrc/mamba/` | Causal conv1d for Mamba models |
| Memory | `csrc/memory/` | Weak reference tensor |
| MoE | `csrc/moe/` | CUTLASS MoE W4A8, alignment, fused gate, top-k softmax/sigmoid, FP8 blockwise MoE |
| Quantization | `csrc/quantization/` | GGUF kernel |
| Speculative | `csrc/speculative/` | EAGLE utils, ngram utils, packbit, speculative sampling |
| Spatial | `csrc/spatial/` | Green context stream |
| Sparse Attention | (Flash Attention) | Sparse flash attention for SM80+ |

### GPU Architecture Support

The kernel library compiles for multiple GPU architectures:

| Architecture | Compute Capability | Flags |
|--------------|--------------------|-------|
| Hopper | SM90, SM90a | Default (fast math) |
| Blackwell | SM100a, SM120a | CUDA 12.8+ |
| Ada Lovelace | SM89 | Optional (ENABLE_BELOW_SM90) |
| Ampere | SM80 | Optional (ENABLE_BELOW_SM90) |

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_BELOW_SM90` | ON (OFF on aarch64) | Enable SM80/SM89 gencode |
| `SGL_KERNEL_ENABLE_BF16` | ON | Enable BF16 support |
| `SGL_KERNEL_ENABLE_FP8` | ON | Enable FP8 support |
| `SGL_KERNEL_ENABLE_FP4` | OFF | Enable FP4 (NVFP4) support |
| `SGL_KERNEL_ENABLE_FA3` | OFF (ON for CUDA >= 12.4) | Enable FlashAttention 3 |
| `SGL_KERNEL_ENABLE_SM90A` | OFF | Enable SM90a |
| `SGL_KERNEL_ENABLE_SM100A` | OFF | Enable SM100a |
| `SGL_KERNEL_COMPILE_THREADS` | 32 | NVCC compilation thread count |

### Third-Party Dependencies

The kernel library integrates these external libraries (fetched at build time):

| Library | Source | Purpose |
|---------|--------|---------|
| CUTLASS | NVIDIA/cutlass | Tensor core GEMM operations |
| fmt | fmtlib/fmt | C++ formatting library |
| Triton | triton-lang/triton v3.5.1 | Triton kernels |
| FlashInfer | flashinfer-ai/flashinfer | Attention kernels |
| sgl-attn | sgl-project/sgl-attn | Sparse flash attention |
| MSCCL++ | microsoft/mscclpp | All-reduce operations |

### Adding New Kernels

Steps to add a new kernel:

1. Implement the kernel in `csrc/`
2. Expose the interface in `include/sgl_kernel_ops.h`
3. Create torch extension in `csrc/common_extension.cc`
4. Update `CMakeLists.txt` to include the new CUDA source
5. Expose Python interface in `python/sgl_kernel/`
6. Add tests and benchmarks

When creating torch extensions, use `m.def` for function definition and `m.impl` for device binding:

```cpp
m.def(
    "bmm_fp8(Tensor A, Tensor B, Tensor! D, Tensor A_scale, Tensor B_scale, "
    "Tensor workspace_buffer, int cublas_handle) -> ()");
m.impl("bmm_fp8", torch::kCUDA, &bmm_fp8);
```

For third-party C++ types, use `make_pytorch_shim` for automatic type conversion.

---

## CUDA Graph Support

SGLang provides three CUDA graph strategies for eliminating kernel launch overhead:

| Strategy | Use Case | Token Shape | Status |
|----------|----------|-------------|--------|
| Standard CUDA Graph | Decode (fixed batch size) | Fixed | Default |
| Breakable CUDA Graph | Debugging, incompatible ops | Fixed | Opt-in |
| Piecewise CUDA Graph | Prefill/Extend (variable tokens) | Variable | Default |

---

## Standard CUDA Graphs

Standard CUDA graphs capture the entire decode forward pass as a single, opaque graph. This works well for decode where the batch size is fixed.

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cuda-graph-max-bs` | Auto | Maximum batch size for CUDA graph capture |
| `--disable-cuda-graph` | `False` | Disable CUDA graph capture entirely |

The graph is captured for batch sizes from 1 to `cuda_graph_max_bs`, with automatic selection of capture sizes (powers of 2, etc.).

---

## Breakable CUDA Graph

Breakable CUDA Graph allows graph breaks to be inserted at specific points, splitting the computation into multiple captured graph segments with eager execution in between.

### Motivation

1. **Debugging**: When something goes wrong inside a captured graph, there is no way to step through operations or insert print statements.
2. **Incompatible ops**: Certain operations (dynamic control flow, host-device sync, JIT compilation) cannot be captured into a CUDA graph.

### Usage

#### Debug Mode (Run Everything Eagerly)

```bash
python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --debug-cuda-graph
```

This wraps the entire decode forward pass in a graph break, so every operation runs eagerly while still going through the full CUDA graph capture/replay code path.

#### Selective Graph Breaks in Model Code

```python
from sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph import eager_on_graph

@eager_on_graph(enable=True)
def my_dynamic_op(x):
    # This op is incompatible with CUDA graph capture
    return some_dynamic_operation(x)
```

#### Bare Graph Break

```python
from sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph import break_graph

def forward(self, x):
    x = self.layer1(x)
    break_graph()  # Force a segment split here
    x = self.layer2(x)
    return x
```

#### Environment Variable

```bash
export SGLANG_USE_BREAKABLE_CUDA_GRAPH=1
python -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct
```

### Server Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--debug-cuda-graph` | `False` | Enable debug/eager mode. Wraps entire forward pass in a graph break. |
| `SGLANG_USE_BREAKABLE_CUDA_GRAPH` | `0` | Enable breakable CUDA graph without debug mode. Required for `@eager_on_graph` decorators. |

### How It Works

**Capture**:
```
Begin capture (segment 1)
  ... graphable ops ...
  @eager_on_graph function encountered:
    1. End current capture segment
    2. Run the function eagerly (allocates output tensors)
    3. Record the function for later replay
    4. Begin new capture segment
  ... more graphable ops ...
End capture (segment N)
```

**Replay**:
```
For each segment i:
  1. Launch CUDA graph segment i
  2. Run the recorded non-graph function i eagerly
Launch final CUDA graph segment
```

**Output Writeback**: When a non-graph function produces output during replay, the result is written back into the same tensor buffers:
- Plain tensors: In-place `copy_()` into original buffer
- Structured outputs (dataclasses, objects): Tensor fields copied in-place
- Dicts of tensors: Tensor values copied in-place

**Stream Fork/Join**: Some models fork work onto secondary CUDA streams. Breakable CUDA graph hooks `torch.cuda.Stream.wait_stream` to track forked streams. When a graph break occurs, all forked streams are automatically joined before ending the capture segment.

### Compatibility

- **NVIDIA CUDA only** (not supported on ROCm/HIP or other platforms)
- **Requires `cuda-python`** (`pip install cuda-python`)
- **Not compatible with memory saver mode** (`SGLANG_MEMORY_SAVER_CUDA_GRAPH`)

### Performance

When no graph breaks are inserted, breakable CUDA graph has minimal overhead. Each graph break adds:
- One `cudaGraphLaunch` call
- One eager Python function call
- One `cudaStreamBeginCapture` / `cudaStreamEndCapture` pair during capture

---

## Piecewise CUDA Graph

Piecewise CUDA Graph (PCG) splits the model's computation graph into pieces (roughly one per layer) at "split points" (e.g., MoE dispatch ops). Each piece is captured as a separate CUDA graph for pre-defined token lengths.

**PCG is enabled by default** for supported configurations. The old `--enable-piecewise-cuda-graph` flag is deprecated.

### Motivation

Standard CUDA graphs capture the entire forward pass as a single graph. This works well for decode (fixed batch size) but not for extend/prefill where token counts vary. PCG solves this by splitting into pieces and capturing each piece separately for multiple token lengths.

### Usage

PCG is enabled by default. No extra flags needed:

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct
```

#### Disable PCG

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --disable-piecewise-cuda-graph
```

#### Custom Capture Sizes

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --piecewise-cuda-graph-max-tokens 2048
```

### Server Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--disable-piecewise-cuda-graph` | `False` | Disable PCG for extend/prefill |
| `--enforce-piecewise-cuda-graph` | `False` | Force-enable PCG, skipping all auto-disable conditions (testing only) |
| `--piecewise-cuda-graph-max-tokens` | `None` (auto) | Maximum token count to capture. Defaults to `chunked_prefill_size` or `2048` (MLA). |
| `--piecewise-cuda-graph-tokens` | `None` (auto) | Explicit list of token lengths to capture |
| `--piecewise-cuda-graph-compiler` | `"eager"` | Compiler backend: `eager` or `inductor` |
| `--enable-piecewise-cuda-graph` | Deprecated | PCG is now enabled by default |

### Shape Configuration

Default capture schedule with increasing granularity:

| Token Range | Step Size |
|-------------|-----------|
| 4 - 32 | 4 |
| 48 - 256 | 16 |
| 288 - 512 | 32 |
| 576 - 1024 | 64 |
| 1280 - 4096 | 256 |
| 4096+ | 512 |

At runtime, the actual token count is rounded up to the nearest captured size (via binary search). If it exceeds the largest captured size, the runtime falls back to the normal forward path.

### Memory Optimization

- **Shared memory pool**: A global shared memory pool is reused across all CUDA graph runners and capture sizes
- **Reverse order capture**: Capture is done from large to small, so smaller graphs reuse memory from larger ones
- **Weak references**: Output tensors of the last subgraph are stored as weak references for maximum reuse
- **Non-torch memory**: The main overhead comes from CUDA graph objects themselves, which scale with the number of captured sizes

### Auto-Disable Conditions

PCG is automatically disabled for:
- Disabled model architectures (e.g., `DeepseekV32ForCausalLM`)
- Speculative decoding
- DP attention
- Pipeline parallelism (`pp_size > 1`)
- Non-CUDA hardware (AMD ROCm, Ascend NPU)
- MoE A2A backend
- LoRA
- Multimodal / VLM models
- DLLM (diffusion LLM)
- Deterministic inference
- PD disaggregation
- Expert distribution recorder / EPLB

Use `--enforce-piecewise-cuda-graph` to skip all checks (testing only).

### Making Kernels PCG-Compatible

New CUDA kernels need to be registered as custom ops for PCG compatibility:

```python
from sglang.srt.utils.custom_op import register_custom_op

# Inplace operator
@register_custom_op(mutates_args=["output_q", "output_s"])
def per_token_group_quant_8bit(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:
    # kernel implementation ...

# Operator with output
@register_custom_op(mutates_args=["x"], out_shape=0)
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x.add_(y)
```

For external library functions (e.g., FlashInfer), use `register_custom_op_from_extern`.

---

## torch.compile Integration

SGLang integrates with PyTorch's `torch.compile` framework for operator fusion and autotuning.

### Enabling torch.compile

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enable-torch-compile
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--enable-torch-compile` | `False` | Enable torch.compile for the model |
| `--torch-compile-max-bs` | `None` | Maximum batch size for torch.compile |

### torch.compile Cache

SGLang uses `max-autotune-no-cudagraphs` mode of torch.compile. The auto-tuning can be slow. For deploying on many machines, ship the compile cache:

```bash
# Step 1: Generate cache
TORCHINDUCTOR_CACHE_DIR=/root/inductor_root_cache python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enable-torch-compile

# Step 2: Copy cache folder to other machines
# Step 3: Launch with cache
TORCHINDUCTOR_CACHE_DIR=/root/inductor_root_cache python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --enable-torch-compile
```

---

## Piecewise CUDA Graph Compiler

The PCG compiler uses `torch.compile` with a custom backend (`SGLangBackend`) to split and compile the model's forward pass.

### Compilation Flow

```
model.forward wrapper
  -> torch.compile(..., backend=SGLangBackend)
  -> FX graph
  -> split_graph() at registered split ops
  -> split_gm (top-level graph that chains the pieces)
  -> replace capturable submodules with CUDAPiecewiseBackend
  -> runtime dispatch: eager split ops + per-piece capture/replay
```

### Stages

1. **Install**: `install_torch_compiled()` replaces `model.forward` with a wrapper. When `is_in_piecewise_cuda_graph()` returns True, the wrapper dispatches to the compiled callable.

2. **Split**: `SGLangBackend` receives the FX graph and calls `split_graph()`. Ops in `CompilationConfig.split_ops` are treated as split points. Split-op submodules run eagerly; surrounding submodules are compiled and wrapped.

3. **Replace**: `PiecewiseCompileInterpreter` iterates over each capturable submodule, compiles it for general (dynamic) shapes, and replaces it with a `CUDAPiecewiseBackend` instance.

4. **Dispatch**: At runtime, split-op submodules run eagerly. Each `CUDAPiecewiseBackend` goes through:
   - **Compile warmup**: Runs the general-shape compiled path
   - **Capture**: For each capture size, runs warmup pass then records a CUDA graph
   - **Steady-state replay**: Replays the captured CUDA graph

### Runner Lifecycle

`PiecewiseCudaGraphRunner` orchestrates:
- **Compile**: Warm up JIT kernels, wrap model with `torch.compile`
- **Capture**: Iterate capture sizes in reverse order (largest first)
- **Replay**: Binary search for nearest captured size, copy inputs, replay graphs

---

## Compilation Pipeline

### Compilation Source Files

| File | Description |
|------|-------------|
| `compilation/backend.py` | `SGLangBackend`, graph splitting, piecewise compilation |
| `compilation/compile.py` | `install_torch_compiled` trampoline |
| `compilation/compilation_config.py` | Capture sizes, split ops, compiler config |
| `compilation/compilation_counter.py` | Compilation metrics and counters |
| `compilation/compiler_interface.py` | Public compiler interface |
| `compilation/cuda_piecewise_backend.py` | Per-subgraph CUDA graph capture/replay |
| `compilation/fix_functionalization.py` | Fix functionalization issues in compiled graphs |
| `compilation/fx_utils.py` | FX graph utility functions |
| `compilation/inductor_pass.py` | Inductor pass for custom optimizations |
| `compilation/npu_piecewise_backend.py` | NPU-specific piecewise backend |
| `compilation/pass_manager.py` | Pass manager for compilation pipeline |
| `compilation/piecewise_context_manager.py` | Global context flags and `ForwardContext` |
| `compilation/weak_ref_tensor.py` | Weak reference tensor for memory optimization |

---

## DeepGEMM Kernels

DeepGEMM provides JIT-compiled FP8 and FP4 GEMM kernels optimized for Hopper (SM90) and Blackwell (SM100) architectures. DeepGEMM is used as a GEMM backend for quantized inference.

### Selection

DeepGEMM is auto-selected when:
- Hardware is SM90 or SM100
- DeepGEMM is installed in the environment
- `--fp8-gemm-backend auto` (default) or `--fp8-gemm-backend deep_gemm`

### FlashInfer DeepGEMM

The `flashinfer_deepgemm` backend uses swapAB optimization for small M dimensions, making it particularly effective for decode-phase GEMMs.

---

## FlashInfer Integration

FlashInfer provides high-performance attention kernels and is integrated into sgl-kernel at build time.

### Attention Backends Using FlashInfer

| Backend | Use Case |
|---------|----------|
| `flashinfer` | Default attention backend for many models |
| `flashinfer_trtllm` | TensorRT-LLM style attention for SM100+ |
| `flashinfer_cutlass` | CUTLASS-based FlashInfer for FP8/FP4 GEMM |
| `flashinfer_cudnn` | cuDNN-based FlashInfer for FP4 on SM120 |
| `flashinfer_deepgemm` | DeepGEMM with swapAB for decode |

### FlashInfer in sgl-kernel

FlashInfer is fetched at build time and compiled into the kernel library. The integration includes norm and renorm kernels from FlashInfer's csrc.

---

## Triton Kernels

SGLang uses Triton kernels as a fallback and for platform-agnostic operations.

### Triton GEMM Backend

The `triton` backend for FP8 GEMM is the universal fallback, working on all platforms:
- NVIDIA: SM80, SM90, SM100, SM120
- AMD: MI300X, MI325X, MI350X

### Triton for Quantized Operations

- `blockwise_int8`: Triton-based blockwise INT8 quantization
- AWQ dequantize on AMD: Triton-based
- GPTQ on AMD: Triton or vLLM kernels

### Triton Kernels in sgl-kernel

Triton kernels (v3.5.1) are fetched at build time and installed as part of the kernel package.

---

## CUTLASS Integration

CUTLASS provides high-performance tensor core GEMM operations and is integrated at multiple levels:

### CUTLASS in sgl-kernel

CUTLASS is fetched at build time (specific commit: `57e3cfb`) and used for:
- FP8 blockwise GEMM kernels
- FP8 GEMM kernel
- INT8 GEMM kernel
- MoE CUTLASS W4A8 kernels
- MLA (Multi-head Latent Attention) kernel

### CUTLASS Backend

The `cutlass` GEMM backend uses sgl-kernel's CUTLASS implementation for:
- FP8 GEMM on SM90/SM100/120
- FP4 GEMM on SM100/120

### Build Flags

```cmake
"-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"
"-DCUTE_USE_PACKED_TUPLE=1"
"-DCUTLASS_DEBUG_TRACE_LEVEL=0"
```

---

## JIT Kernel Compilation

Some kernels in SGLang are compiled just-in-time at runtime rather than pre-compiled:

### DeepGEMM

DeepGEMM kernels are JIT-compiled when first used. This allows them to adapt to the specific hardware and model configuration.

### Triton Kernels

Triton kernels are also JIT-compiled, with caching to avoid recompilation on subsequent runs.

### torch.compile Auto-Tuning

When `--enable-torch-compile` is used, PyTorch's inductor performs auto-tuning to find optimal kernel configurations. This process can be slow on first run but is cached for subsequent deployments.

---

## Performance Optimization Tips

### Memory Optimization

1. **Reduce CUDA graph captures**: Use `--cuda-graph-max-bs` to limit the number of captured graphs
2. **Limit PCG capture sizes**: Use `--piecewise-cuda-graph-max-tokens` to cap the maximum captured token count
3. **Memory saver mode**: Use `SGLANG_MEMORY_SAVER_CUDA_GRAPH` for memory-constrained environments (incompatible with breakable CUDA graph)

### Throughput Optimization

1. **Enable PCG**: Piecewise CUDA Graph is enabled by default and reduces prefill kernel launch overhead
2. **Use torch.compile**: Can provide operator fusion benefits (benchmark to verify)
3. **Cache torch.compile results**: Ship `TORCHINDUCTOR_CACHE_DIR` across machines
4. **Choose optimal GEMM backend**: DeepGEMM for SM90/SM100, FlashInfer TRTLLM for SM100, Aiter for AMD

### Debugging Tips

1. **Breakable CUDA Graph**: Use `--debug-cuda-graph` to run all ops eagerly for debugging
2. **Disable PCG**: Use `--disable-piecewise-cuda-graph` if encountering PCG-related errors
3. **Kernel analysis**: Use `analyze_whl_kernel_sizes.py` to identify oversized kernels

### Bug Reporting for PCG

If you encounter PCG errors during startup:
1. Add `--disable-piecewise-cuda-graph` to work around the issue
2. Report the bug with: full traceback, model name, quantization method, launch command, GPU type and driver version

---

## Source Code Structure

### sgl-kernel Directory

| Directory/File | Description |
|----------------|-------------|
| `sgl-kernel/csrc/` | CUDA/C++ kernel implementations |
| `sgl-kernel/csrc/allreduce/` | Custom all-reduce kernels |
| `sgl-kernel/csrc/attention/` | Attention kernels |
| `sgl-kernel/csrc/elementwise/` | Element-wise operation kernels |
| `sgl-kernel/csrc/expert_specialization/` | Expert specialization kernels |
| `sgl-kernel/csrc/gemm/` | GEMM kernels (FP8, INT8, AWQ, GPTQ, etc.) |
| `sgl-kernel/csrc/grammar/` | Grammar constraint kernels |
| `sgl-kernel/csrc/kvcacheio/` | KV cache I/O operations |
| `sgl-kernel/csrc/mamba/` | Mamba/causal conv kernels |
| `sgl-kernel/csrc/moe/` | MoE kernels |
| `sgl-kernel/csrc/quantization/` | Quantization kernels (GGUF) |
| `sgl-kernel/csrc/speculative/` | Speculative decoding kernels |
| `sgl-kernel/csrc/spatial/` | Spatial/green context kernels |
| `sgl-kernel/include/` | C++ header files |
| `sgl-kernel/python/sgl_kernel/` | Python bindings |
| `sgl-kernel/tests/` | Test suite |
| `sgl-kernel/benchmark/` | Benchmark suite |
| `sgl-kernel/CMakeLists.txt` | Build configuration |
| `sgl-kernel/README.md` | Documentation |

### Compilation Pipeline

| File | Description |
|------|-------------|
| `python/sglang/srt/compilation/backend.py` | SGLang torch.compile backend |
| `python/sglang/srt/compilation/compile.py` | Install torch compiled wrapper |
| `python/sglang/srt/compilation/compilation_config.py` | Compilation configuration |
| `python/sglang/srt/compilation/compilation_counter.py` | Compilation metrics |
| `python/sglang/srt/compilation/compiler_interface.py` | Public compiler interface |
| `python/sglang/srt/compilation/cuda_piecewise_backend.py` | CUDA piecewise graph backend |
| `python/sglang/srt/compilation/npu_piecewise_backend.py` | NPU piecewise backend |
| `python/sglang/srt/compilation/fix_functionalization.py` | Functionalization fixes |
| `python/sglang/srt/compilation/fx_utils.py` | FX graph utilities |
| `python/sglang/srt/compilation/inductor_pass.py` | Inductor passes |
| `python/sglang/srt/compilation/pass_manager.py` | Pass management |
| `python/sglang/srt/compilation/piecewise_context_manager.py` | Context management |
| `python/sglang/srt/compilation/weak_ref_tensor.py` | Weak reference tensor |

### CUDA Graph Runners

| File | Description |
|------|-------------|
| `python/sglang/srt/model_executor/cuda_graph_runner.py` | Standard CUDA graph runner |
| `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` | Piecewise CUDA graph runner |
| `python/sglang/srt/model_executor/breakable_cuda_graph/breakable_cuda_graph.py` | Breakable CUDA graph core |
| `python/sglang/srt/model_executor/breakable_cuda_graph/cuda_utils.py` | CUDA runtime bindings |
| `python/sglang/srt/utils/custom_op.py` | Custom op registration for PCG |

---

## References

- [sgl-kernel GitHub](https://github.com/sgl-project/sglang/tree/main/sgl-kernel) - Kernel library source
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) - FP8/FP4 GEMM kernels
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) - Attention kernel library
- [CUTLASS](https://github.com/NVIDIA/cutlass) - NVIDIA tensor core GEMM
- [Triton](https://github.com/triton-lang/triton) - JIT kernel compilation
- [torch.compile Cache Tutorial](https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)
- [CUDA Graphs Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
