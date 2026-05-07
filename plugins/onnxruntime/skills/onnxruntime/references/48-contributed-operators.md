# ONNX Runtime Reference - Chapter 48: Contributed Operators

This chapter covers ONNX Runtime's contributed operators (contrib ops), which are non-standard ONNX operators that provide optimized implementations for specific hardware targets and common model patterns.

---

## 48.1 What Are Contrib Ops

### 48.1.1 Definition

Contributed operators (contrib ops) are custom operators that are not part of the standard ONNX specification. They are implemented within ONNX Runtime to:

1. **Optimize common patterns**: Fuse multi-node patterns into single efficient kernels
2. **Support proprietary extensions**: Microsoft-specific or hardware-specific operations
3. **Enable new functionality**: Operations not yet standardized in ONNX
4. **Improve performance**: Hardware-accelerated implementations for specific targets

### 48.1.2 Contrib Op Domains

```
com.microsoft     - Microsoft-contributed operators (primary contrib domain)
ai.onnx.contrib   - Reserved for community contributions
```

### 48.1.3 Contrib vs Standard ONNX Ops

| Aspect | Standard ONNX Ops | Contrib Ops |
|--------|------------------|-------------|
| Domain | "" (empty) or "ai.onnx" | "com.microsoft" |
| Specification | ONNX standard | ORT-specific |
| Portability | All ONNX runtimes | ONNX Runtime only |
| Optimization | General purpose | Hardware-optimized |
| Versioning | Opset version | Independent versioning |

---

## 48.2 contrib_ops/cpu/ Directory Structure and Operators

### 48.2.1 Directory Layout

```
onnxruntime/contrib_ops/cpu/
├── contrib_ops.pc                        # Build configuration
├── cpu_contrib_kernels.cc                # Kernel registration
├── attention.h                           # Attention operator
├── attention.cc
├── attention_base.h                      # Base class for attention
├── attention_base.cc
├── attention_common.h                    # Common attention utilities
├── beam_search.h                         # Beam search decoding
├── beam_search.cc
├── beam_search_parameters.h
├── bias_dropout.h                        # Bias + Dropout fusion
├── bias_dropout.cc
├── bias_gelu.h                           # Bias + GELU fusion
├── bias_gelu.cc
├── bias_split_gelu.h                     # Bias + Split + GELU
├── bias_split_gelu.cc
├── bitmask_dropout.h                     # Bitmask-based dropout
├── bitmask_dropout.cc
├── crop_and_resize.h                     # Crop and resize
├── crop_and_resize.cc
├── dynamic_quantize_matmul.h             # Dynamic quantized MatMul
├── dynamic_quantize_matmul.cc
├── dynamic_slice.h                       # Dynamic slice (legacy)
├── dynamic_slice.cc
├── embed_layer_norm.h                    # Embedding + LayerNorm fusion
├── embed_layer_norm.cc
├── expand_dims.h                         # Expand dimensions
├── expand_dims.cc
├── fast_gelu.h                           # Fast GELU approximation
├── fast_gelu.cc
├── function_ops/                         # Function-based contrib ops
│   ├── function_ops.cc
│   └── ...
├── fused_conv.h                          # Fused Convolution
├── fused_conv.cc
├── fused_gemm.h                          # Fused GEMM
├── fused_gemm.cc
├── fused_matmul.h                        # Fused MatMul
├── fused_matmul.cc
├── gather_elements.h                     # GatherElements (legacy)
├── gather_elements.cc
├── gelu.h                                # GELU activation
├── gelu.cc
├── grid_sample.h                         # Grid sample
├── grid_sample.cc
├── inverse.h                             # Matrix inverse
├── inverse.cc
├── layer_norm.h                          # Layer normalization
├── layer_norm.cc
├── layer_norm_impl/                      # LayerNorm implementation
│   ├── layer_norm_impl.h
│   ├── layer_norm_base.h
│   └── ...
├── longformer_attention.h                # Longformer attention
├── longformer_attention.cc
├── matmul_nbits.h                        # MatMul with N-bit quantization
├── matmul_nbits.cc
├── matmul_integer_to_float.h             # Integer MatMul to float output
├── matmul_integer_to_float.cc
├── matmul_sparse.h                       # Sparse MatMul
├── matmul_sparse.cc
├── multihead_attention.h                 # Multi-head attention
├── multihead_attention.cc
├── murmur_hash3.h                        # MurmurHash3
├── murmur_hash3.cc
├── nchwc_ops.h                           # NCHWc layout operations
├── nchwc_ops.cc
├── no_input_ops.h                        # Ops with no inputs
├── no_input_ops.cc
├── nv_embedding.h                        # NV embedding
├── nv_embedding.cc
├── pnhwc_ops.h                           # Packed NHWC operations
├── pnhwc_ops.cc
├── qlinear_concat.h                      # Quantized linear concat
├── qlinear_concat.cc
├── qlinear_global_average_pool.h         # Quantized global average pool
├── qlinear_global_average_pool.cc
├── qlinear_lookup_table.h                # Quantized lookup table
├── qlinear_lookup_table.cc
├── qlinear_pool.h                        # Quantized pool
├── qlinear_pool.cc
├── qlinear_residue.h                     # Quantized residue ops
├── qlinear_residue.cc
├── quantize/                             # Quantization contrib ops
│   ├── matmul_nbits_quantize.h
│   ├── matmul_nbits_quantize.cc
│   └── ...
├── range.h                               # Range op
├── range.cc
├── regex_split.h                         # Regex split
├── regex_split.cc
├── regex_split_with_offsets.h
├── regex_split_with_offsets.cc
├── relative_attention_bias.h             # Relative attention bias
├── relative_attention_bias.cc
├── sample_op.h                           # Sampling ops
├── sample_op.cc
├── sentinel_ops.h                        # Sentinel operators
├── sentinel_ops.cc
├── skip_layer_norm.h                     # Skip connection + LayerNorm
├── skip_layer_norm.cc
├── skip_simplified_layer_norm.h          # Skip + simplified LayerNorm
├── skip_simplified_layer_norm.cc
├── snpe/                                 # SNPE backend ops
│   └── ...
├── tokenizer.h                           # Tokenizer
├── tokenizer.cc
├── torch_embedding.h                     # Torch-style embedding
├── torch_embedding.cc
├── trilu.h                               # Triangular utility
├── trilu.cc
├── unique.h                              # Unique elements
├── unique.cc
├── vas/                                  # Video Analytics Suite ops
│   └── ...
├── vocabs.h                              # Vocabulary ops
├── vocabs.cc
├── watchdog.h                            # Watchdog timer
├── watchdog.cc
├── word_embedding.h                      # Word embedding
├── word_embedding.cc
└── maxml.h                               # MaxML ops
    └── maxml.cc
```

### 48.2.2 Key CPU Contrib Operators

#### Attention Operator

```cpp
// contrib_ops/cpu/attention.h
// Domain: com.microsoft
// Fuses: MatMul(Q) + MatMul(K) + MatMul(V) + Softmax + MatMul
//
// Inputs:
//   input (T): [batch_size, sequence_length, hidden_size]
//   weights (T): [hidden_size, 3 * hidden_size]  (QKV packed)
//   bias (T): [3 * hidden_size]
//   (optional) mask_index (M): attention mask
//   (optional) past (T): past key/value states
//   (optional) extra_add (T): additional attention bias
//
// Outputs:
//   output (T): [batch_size, sequence_length, hidden_size]
//   present (T): present key/value states (for autoregressive)
//
// Attributes:
//   num_heads (int): Number of attention heads
//   unidirectional (int): 1 for causal, 0 for bidirectional
//   scale (float): Attention scale factor (default: 1/sqrt(d))
//
// Type constraints:
//   T: float, float16, bfloat16
//   M: int32

class Attention : public OpKernel {
public:
    Attention(const OpKernelInfo& info);
    Status Compute(OpKernelContext* context) const override;

private:
    int num_heads_;
    bool is_unidirectional_;
    float scale_;
    bool do_rotary_;
    int rotary_embedding_dim_;
    bool past_present_share_buffer_;
    int kv_num_heads_;  // For GQA (Grouped Query Attention)
};
```

#### EmbedLayerNormalization Operator

```cpp
// contrib_ops/cpu/embed_layer_norm.h
// Domain: com.microsoft
// Fuses: WordEmbedding + PositionEmbedding + SegmentEmbedding + LayerNorm
//
// Inputs:
//   input_ids (int32): [batch_size, sequence_length]
//   (optional) position_ids (int32): [batch_size, sequence_length]
//   (optional) segment_ids (int32): [batch_size, sequence_length]
//   word_embedding (T): [vocab_size, hidden_size]
//   position_embedding (T): [max_position, hidden_size]
//   (optional) segment_embedding (T): [num_segments, hidden_size]
//   (optional) layer_norm_weight (T): [hidden_size]
//   (optional) layer_norm_bias (T): [hidden_size]
//   (optional) mask (int32): [batch_size, sequence_length]
//   (optional) mask_type (int32): mask type
//
// Outputs:
//   output (T): [batch_size, sequence_length, hidden_size]
//   (optional) mask_index (int32): [batch_size]
//   (optional) embedding_sum (T): for debugging

class EmbedLayerNormalization : public OpKernel {
public:
    EmbedLayerNormalization(const OpKernelInfo& info);
    Status Compute(OpKernelContext* context) const override;

private:
    float epsilon_;
    int64_t layer_norm_eps_;
};
```

#### SkipSimplifiedLayerNormalization Operator

```cpp
// contrib_ops/cpu/skip_simplified_layer_norm.h
// Domain: com.microsoft
// Fuses: SkipConnection + LayerNorm (simplified, no mean subtraction)
//
// Simplified LayerNorm: output = input * gamma / RMS(input)
// Where RMS(input) = sqrt(mean(input^2) + epsilon)
//
// Inputs:
//   input (T): [batch_size, seq_len, hidden_size]
//   skip (T): [batch_size, seq_len, hidden_size] (skip connection)
//   gamma (T): [hidden_size]
//   (optional) beta (T): [hidden_size]
//   (optional) bias (T): [hidden_size]
//
// Outputs:
//   output (T): [batch_size, seq_len, hidden_size]

class SkipSimplifiedLayerNormalization : public OpKernel {
public:
    SkipSimplifiedLayerNormalization(const OpKernelInfo& info);
    Status Compute(OpKernelContext* context) const override;

private:
    float epsilon_;
};
```

#### BiasGelu Operator

```cpp
// contrib_ops/cpu/bias_gelu.h
// Domain: com.microsoft
// Fuses: BiasAdd + GELU activation
//
// GELU(x) = x * Phi(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
// Fast approximation: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Inputs:
//   A (T): [any_shape] (input tensor)
//   B (T): [last_dim] (bias, broadcasted)
//
// Outputs:
//   output (T): [same as A]

class BiasGelu : public OpKernel {
public:
    BiasGelu(const OpKernelInfo& info);
    Status Compute(OpKernelContext* context) const override;
};
```

#### MatMulNBits Operator

```cpp
// contrib_ops/cpu/matmul_nbits.h
// Domain: com.microsoft
// Matrix multiplication with N-bit quantized weights (4-bit, 8-bit)
//
// Supports:
//   - 4-bit quantization (NF4, FP4, INT4)
//   - 8-bit quantization
//   - Per-channel scaling
//   - Block-wise quantization
//
// Inputs:
//   A (float): [M, K]  (activations, float)
//   B (uint8): [K/2, N] (4-bit packed weights)
//   scales (float): [N] or [K/block_size, N] (per-channel or per-block)
//   zero_points (uint8): [N] or [K/block_size, N] (optional)
//   (optional) bias (float): [N]
//
// Outputs:
//   Y (float): [M, N]

class MatMulNBits : public OpKernel {
public:
    MatMulNBits(const OpKernelInfo& info);
    Status Compute(OpKernelContext* context) const override;

private:
    int64_t n_bits_;          // Number of bits (4 or 8)
    int64_t block_size_;      // Quantization block size (default: 128)
    int64_t group_size_;      // Group size for scaling
    bool is_4bit_;
};
```

#### MultiHeadAttention Operator

```cpp
// contrib_ops/cpu/multihead_attention.h
// Domain: com.microsoft
// Full multi-head attention with separate Q, K, V inputs
//
// Inputs:
//   Q (T): [batch_size, seq_len_q, head_size] or [batch, num_heads, seq_q, head_size]
//   K (T): [batch_size, seq_len_k, head_size] or [batch, num_heads, seq_k, head_size]
//   V (T): [batch_size, seq_len_k, head_size]
//   (optional) bias (T): [num_heads] or [1]
//   (optional) key_padding_mask (T): [batch_size, seq_len_k]
//   (optional) attention_mask (T): [seq_len_q, seq_len_k] or [batch, seq_q, seq_k]
//   (optional) past_k (T): [batch, num_heads, past_seq, head_size]
//   (optional) past_v (T): [batch, num_heads, past_seq, head_size]
//
// Outputs:
//   output (T): [batch_size, seq_len_q, num_heads * head_size]

class MultiHeadAttention : public OpKernel {
public:
    MultiHeadAttention(const OpKernelInfo& info);
    Status Compute(OpKernelContext* context) const override;

private:
    int num_heads_;
    bool is_unidirectional_;
    float scale_;
    int kv_num_heads_;
};
```

#### GridSample Operator

```cpp
// contrib_ops/cpu/grid_sample.h
// Domain: com.microsoft
// Samples from input using a spatial transformation grid
//
// Inputs:
//   input (T): [N, C, H_in, W_in]
//   grid (T): [N, H_out, W_out, 2]
//
// Attributes:
//   mode (string): "bilinear" or "nearest"
//   padding_mode (string): "zeros", "border", or "reflection"
//   align_corners (int): 0 or 1

class GridSample : public OpKernel {
public:
    GridSample(const OpKernelInfo& info);
    Status Compute(OpKernelContext* context) const override;

private:
    std::string mode_;
    std::string padding_mode_;
    bool align_corners_;
};
```

#### QLinearConcat Operator

```cpp
// contrib_ops/cpu/qlinear_concat.h
// Domain: com.microsoft
// Quantized concatenation of tensors
//
// Inputs:
//   (repeating) input (uint8/int8): quantized input
//   (repeating) input_scale (float): input quantization scale
//   (repeating) input_zp (int): input zero point
//   output_scale (float): output quantization scale
//   output_zp (int): output zero point
//   axis (int): concatenation axis

class QLinearConcat : public OpKernel {
public:
    QLinearConcat(const OpKernelInfo& info);
    Status Compute(OpKernelContext* context) const override;
};
```

#### Tokenizer Operator

```cpp
// contrib_ops/cpu/tokenizer.h
// Domain: com.microsoft
// Text tokenization
//
// Inputs:
//   text (string): Input text to tokenize
//
// Attributes:
//   tokenizer_class (string): Tokenizer class name
//   tokenizer_json (string): JSON tokenizer configuration
//   vocab_file (string): Vocabulary file path
//   merges_file (string): BPE merges file path
//   pad_token_name (string): Padding token
//   eos_token_name (string): End of sequence token
//   bos_token_name (string): Beginning of sequence token
//   pad_value (int): Padding value (default: 0)
//   token_out_id (int): Output token IDs or strings (default: 1 = IDs)

class Tokenizer : public OpKernel {
public:
    Tokenizer(const OpKernelInfo& info);
    Status Compute(OpKernelContext* context) const override;

private:
    std::string tokenizer_class_;
    std::string vocab_file_;
    std::string merges_file_;
    // ... tokenizer state
};
```

#### WordEmbedding Operator

```cpp
// contrib_ops/cpu/word_embedding.h
// Domain: com.microsoft
// Word embedding lookup with optional position and segment embeddings
//
// Inputs:
//   input_ids (int32): Token IDs [batch_size, seq_len]
//   word_embeddings (T): [vocab_size, hidden_size]
//   (optional) position_ids (int32): [batch_size, seq_len]
//   (optional) position_embeddings (T): [max_position, hidden_size]
//
// Outputs:
//   output (T): [batch_size, seq_len, hidden_size]

class WordEmbedding : public OpKernel {
public:
    WordEmbedding(const OpKernelInfo& info);
    Status Compute(OpKernelContext* context) const override;
};
```

#### FastGelu Operator

```cpp
// contrib_ops/cpu/fast_gelu.h
// Domain: com.microsoft
// Fast GELU approximation
//
// FastGelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Inputs:
//   X (T): Input tensor
//   (optional) bias (T): Bias (broadcasted)
//
// Outputs:
//   Y (T): Output tensor (same shape as X)

class FastGelu : public OpKernel {
public:
    FastGelu(const OpKernelInfo& info);
    Status Compute(OpKernelContext* context) const override;
};
```

#### NCHWC Operations

```cpp
// contrib_ops/cpu/nchwc_ops.h
// Domain: com.microsoft
// Optimized operations for NCHWc (channel-last blocked) layout
//
// The NCHWc format blocks channels into groups of 4/8/16
// for better SIMD utilization on x86:
//   NCHW [N, C, H, W] → NCHWc [N, C/c, H, W, c]
//
// Operations:
//   - NchwcConv: Convolution in NCHWc layout
//   - NchwcPool: Pooling in NCHWc layout
//   - NchwcReorder: Layout conversion NCHW ↔ NCHWc
//   - NchwcBatchNorm: BatchNorm in NCHWc layout

class NchwcConv : public OpKernel {
public:
    NchwcConv(const OpKernelInfo& info);
    Status Compute(OpKernelContext* context) const override;

private:
    int64_t group_;
    std::vector<int64_t> kernel_shape_;
    std::vector<int64_t> strides_;
    std::vector<int64_t> pads_;
    std::vector<int64_t> dilations_;
    int64_t activation_type_;  // 0=None, 1=ReLU, 2=ClippedReLU, 3=Sigmoid
};
```

---

## 48.3 contrib_ops/cuda/ Operators

### 48.3.1 Directory Layout

```
onnxruntime/contrib_ops/cuda/
├── cuda_contrib_kernels.cc               # Kernel registration
├── attention.h                           # GPU attention
├── attention.cc
├── attention_bias.h                      # Attention bias
├── attention_bias.cc
├── attention_gpu_common.h                # Common GPU attention utilities
├── bias_gelu.h                           # GPU BiasGelu
├── bias_gelu.cc
├── bias_split_gelu.h                     # GPU BiasSplitGelu
├── bias_split_gelu.cc
├── binary_gelu.h                         # GPU BinaryGelu
├── binary_gelu.cc
├── builtin_embedding.h                   # GPU Embedding
├── builtin_embedding.cc
├── dynamic_disjoint_set.h                # Dynamic disjoint set
├── dynamic_disjoint_set.cc
├── dynamic_quantize_matmul.h             # GPU dynamic quantized MatMul
├── dynamic_quantize_matmul.cc
├── embed_layer_norm.h                    # GPU EmbedLayerNorm
├── embed_layer_norm.cc
├── fast_gelu.h                           # GPU FastGelu
├── fast_gelu.cc
├── flash_attention/                      # Flash Attention implementation
│   ├── flash_attention.h
│   ├── flash_attention.cc
│   ├── flash_attention_api.cpp
│   ├── flash_attention_fwd.cc
│   └── flash_attention_bwd.cc
├── fused_conv.h                          # GPU fused Conv
├── fused_conv.cc
├── gelu.h                                # GPU GELU
├── gelu.cc
├── layer_norm.h                          # GPU LayerNorm
├── layer_norm.cc
├── longformer_attention.h                # GPU Longformer attention
├── longformer_attention.cc
├── matmul_nbits.h                        # GPU MatMulNBits
├── matmul_nbits.cc
├── matmul_integer_to_float.h             # GPU MatMulIntegerToFloat
├── matmul_integer_to_float.cc
├── multihead_attention.h                 # GPU MultiHeadAttention
├── multihead_attention.cc
├── ngram_repeat_block.h                  # N-gram repeat blocking
├── ngram_repeat_block.cc
├── nv_embedding.h                        # NV Embedding (GPU)
├── nv_embedding.cc
├── qlinear_concat.h                      # GPU QLinearConcat
├── qlinear_concat.cc
├── skip_layer_norm.h                     # GPU SkipLayerNorm
├── skip_layer_norm.cc
├── skip_simplified_layer_norm.h          # GPU SkipSimplifiedLayerNorm
├── skip_simplified_layer_norm.cc
├── tokenizer.h                           # GPU Tokenizer (limited)
├── tokenizer.cc
├── trig_exp.h                            # Trig/Exp contrib ops
├── trig_exp.cc
├── vas/                                  # VAS GPU ops
│   └── ...
└── watchdog.h                            # GPU watchdog
    └── watchdog.cc
```

### 48.3.2 Key CUDA Contrib Operators

#### CUDA Attention (with Flash Attention)

```cpp
// contrib_ops/cuda/attention.h
// GPU-optimized attention with multiple backends:
// 1. cuBLAS-based (standard)
// 2. Flash Attention (memory-efficient)
// 3. Memory-efficient attention (xformers-style)
//
// Flash Attention reduces memory from O(N^2) to O(N) for attention
// by recomputing attention weights during backward pass

class Attention final : public OpKernel {
public:
    Attention(const OpKernelInfo& info);
    Status ComputeInternal(OpKernelContext* context) const override;

private:
    int num_heads_;
    int kv_num_heads_;  // For GQA
    bool is_unidirectional_;
    float scale_;
    bool use_flash_attention_;
    bool use_memory_efficient_attention_;
    int rotary_embedding_dim_;
    // ... other config
};
```

#### CUDA MatMulNBits

```cpp
// contrib_ops/cuda/matmul_nbits.h
// GPU implementation of N-bit quantized MatMul
//
// Uses:
// - CUDA cores for small matrices
// - Tensor Cores for larger matrices (when available)
// - Block-wise dequantization + FP16 accumulation

class MatMulNBits final : public OpKernel {
public:
    MatMulNBits(const OpKernelInfo& info);
    Status ComputeInternal(OpKernelContext* context) const override;

private:
    int64_t n_bits_;
    int64_t block_size_;
    int64_t group_size_;
    // Pre-computed dequantization parameters
    mutable bool initialized_ = false;
    mutable cudaEvent_t done_event_;
};
```

---

## 48.4 contrib_ops/js/ Operators

### 48.4.1 Overview

The JavaScript contrib ops provide optimized implementations for the WebAssembly and WebGL backends.

```
onnxruntime/contrib_ops/js/
├── js_contrib_kernels.cc
├── fft.h                    # FFT for JS backend
├── fft.cc
├── array_creation.h         # Array creation ops
├── array_creation.cc
└── ...
```

---

## 48.5 contrib_ops/webgpu/ Operators

### 48.5.1 Overview

WebGPU contrib ops provide GPU-accelerated implementations using WGSL compute shaders.

```
onnxruntime/contrib_ops/webgpu/
├── webgpu_contrib_kernels.cc
├── attention.h              # WebGPU attention
├── attention.cc
├── embed_layer_norm.h       # WebGPU EmbedLayerNorm
├── embed_layer_norm.cc
├── skip_layer_norm.h        # WebGPU SkipLayerNorm
├── skip_layer_norm.cc
├── fast_gelu.h              # WebGPU FastGelu
├── fast_gelu.cc
├── bias_gelu.h              # WebGPU BiasGelu
├── bias_gelu.cc
├── matmul_nbits.h           # WebGPU MatMulNBits
├── matmul_nbits.cc
└── shaders/                 # WGSL shader files
    ├── attention.wgsl
    ├── layer_norm.wgsl
    ├── fast_gelu.wgsl
    ├── matmul_nbits.wgsl
    └── ...
```

---

## 48.6 Registration Mechanism

### 48.6.1 CPU Contrib Kernel Registration

```cpp
// contrib_ops/cpu/cpu_contrib_kernels.cc

Status RegisterCPUContribKernels(KernelRegistry& kernel_registry) {
    // Register all CPU contrib kernels
    static const BuildKernelCreateInfoFn kernel_create_fn_list[] = {
        BuildKernelCreateInfo<onnxruntime::contrib::Attention>,
        BuildKernelCreateInfo<onnxruntime::contrib::BiasGelu>,
        BuildKernelCreateInfo<onnxruntime::contrib::BiasDropout>,
        BuildKernelCreateInfo<onnxruntime::contrib::BiasSplitGelu>,
        BuildKernelCreateInfo<onnxruntime::contrib::CropAndResize>,
        BuildKernelCreateInfo<onnxruntime::contrib::DynamicQuantizeMatMul>,
        BuildKernelCreateInfo<onnxruntime::contrib::EmbedLayerNormalization>,
        BuildKernelCreateInfo<onnxruntime::contrib::ExpandDims>,
        BuildKernelCreateInfo<onnxruntime::contrib::FastGelu>,
        BuildKernelCreateInfo<onnxruntime::contrib::FusedConv>,
        BuildKernelCreateInfo<onnxruntime::contrib::FusedGemm>,
        BuildKernelCreateInfo<onnxruntime::contrib::FusedMatMul>,
        BuildKernelCreateInfo<onnxruntime::contrib::Gelu>,
        BuildKernelCreateInfo<onnxruntime::contrib::GridSample>,
        BuildKernelCreateInfo<onnxruntime::contrib::Inverse>,
        BuildKernelCreateInfo<onnxruntime::contrib::LayerNorm>,
        BuildKernelCreateInfo<onnxruntime::contrib::LongformerAttention>,
        BuildKernelCreateInfo<onnxruntime::contrib::MatMulNBits>,
        BuildKernelCreateInfo<onnxruntime::contrib::MatMulIntegerToFloat>,
        BuildKernelCreateInfo<onnxruntime::contrib::MatMulSparse>,
        BuildKernelCreateInfo<onnxruntime::contrib::MultiHeadAttention>,
        BuildKernelCreateInfo<onnxruntime::contrib::MurmurHash3>,
        BuildKernelCreateInfo<onnxruntime::contrib::NchwcConv>,
        BuildKernelCreateInfo<onnxruntime::contrib::NchwcPool>,
        BuildKernelCreateInfo<onnxruntime::contrib::NchwcReorder>,
        BuildKernelCreateInfo<onnxruntime::contrib::NvEmbedding>,
        BuildKernelCreateInfo<onnxruntime::contrib::QLinearConcat>,
        BuildKernelCreateInfo<onnxruntime::contrib::QLinearGlobalAveragePool>,
        BuildKernelCreateInfo<onnxruntime::contrib::QLinearLookupTable>,
        BuildKernelCreateInfo<onnxruntime::contrib::QLinearPool>,
        BuildKernelCreateInfo<onnxruntime::contrib::Range>,
        BuildKernelCreateInfo<onnxruntime::contrib::RegexSplit>,
        BuildKernelCreateInfo<onnxruntime::contrib::RegexSplitWithOffsets>,
        BuildKernelCreateInfo<onnxruntime::contrib::RelativeAttentionBias>,
        BuildKernelCreateInfo<onnxruntime::contrib::SampleOp>,
        BuildKernelCreateInfo<onnxruntime::contrib::SentinelOps>,
        BuildKernelCreateInfo<onnxruntime::contrib::SkipLayerNormalization>,
        BuildKernelCreateInfo<onnxruntime::contrib::SkipSimplifiedLayerNormalization>,
        BuildKernelCreateInfo<onnxruntime::contrib::Tokenizer>,
        BuildKernelCreateInfo<onnxruntime::contrib::TorchEmbedding>,
        BuildKernelCreateInfo<onnxruntime::contrib::Trilu>,
        BuildKernelCreateInfo<onnxruntime::contrib::Unique>,
        BuildKernelCreateInfo<onnxruntime::contrib::WordEmbedding>,
        // ... more kernels
    };

    for (auto& create_fn : kernel_create_fn_list) {
        ORT_RETURN_IF_ERROR(kernel_registry.Register(create_fn()));
    }

    return Status::OK();
}
```

### 48.6.2 CUDA Contrib Kernel Registration

```cpp
// contrib_ops/cuda/cuda_contrib_kernels.cc

Status RegisterCudaContribKernels(KernelRegistry& kernel_registry) {
    static const BuildKernelCreateInfoFn kernel_create_fn_list[] = {
        BuildKernelCreateInfo<onnxruntime::contrib::cuda::Attention>,
        BuildKernelCreateInfo<onnxruntime::contrib::cuda::BiasGelu>,
        BuildKernelCreateInfo<onnxruntime::contrib::cuda::BiasSplitGelu>,
        BuildKernelCreateInfo<onnxruntime::contrib::cuda::EmbedLayerNormalization>,
        BuildKernelCreateInfo<onnxruntime::contrib::cuda::FastGelu>,
        BuildKernelCreateInfo<onnxruntime::contrib::cuda::FusedConv>,
        BuildKernelCreateInfo<onnxruntime::contrib::cuda::Gelu>,
        BuildKernelCreateInfo<onnxruntime::contrib::cuda::LayerNorm>,
        BuildKernelCreateInfo<onnxruntime::contrib::cuda::LongformerAttention>,
        BuildKernelCreateInfo<onnxruntime::contrib::cuda::MatMulNBits>,
        BuildKernelCreateInfo<onnxruntime::contrib::cuda::MatMulIntegerToFloat>,
        BuildKernelCreateInfo<onnxruntime::contrib::cuda::MultiHeadAttention>,
        BuildKernelCreateInfo<onnxruntime::contrib::cuda::QLinearConcat>,
        BuildKernelCreateInfo<onnxruntime::contrib::cuda::SkipLayerNormalization>,
        BuildKernelCreateInfo<onnxruntime::contrib::cuda::SkipSimplifiedLayerNormalization>,
        // ... more CUDA contrib kernels
    };

    for (auto& create_fn : kernel_create_fn_list) {
        ORT_RETURN_IF_ERROR(kernel_registry.Register(create_fn()));
    }

    return Status::OK();
}
```

### 48.6.3 BuildKernelCreateInfo Function

```cpp
// Each contrib op must define a BuildKernelCreateInfo specialization
template <typename TOpKernel>
std::unique_ptr<KernelCreateInfo> BuildKernelCreateInfo() {
    // Use the kernel's static CreateInfo() method
    return TOpKernel::GetKernelCreateInfo();
}

// Example: Attention kernel's info
class Attention : public OpKernel {
public:
    static std::unique_ptr<KernelCreateInfo> GetKernelCreateInfo() {
        auto def = KernelDefBuilder()
            .SetName("Attention")
            .SetDomain("com.microsoft")
            .SinceVersion(1)
            .Provider(kCpuExecutionProvider)
            .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                   DataTypeImpl::GetTensorType<MLFloat16>(),
                                   DataTypeImpl::GetTensorType<BFloat16>()})
            .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>())
            .Build();

        return std::make_unique<KernelCreateInfo>(
            std::move(def),
            [](const OpKernelInfo& info) -> std::unique_ptr<OpKernel> {
                return std::make_unique<Attention>(info);
            });
    }
};
```

### 48.6.4 Kernel Registration During Session Creation

```cpp
// onnxruntime/core/session/inference_session.cc
Status InferenceSession::Initialize() {
    // Register standard ONNX kernels
    ORT_RETURN_IF_ERROR(RegisterKernels(kernel_registry_));

    // Register contrib kernels
    ORT_RETURN_IF_ERROR(RegisterCPUContribKernels(kernel_registry_));

    // Register EP-specific contrib kernels
    for (const auto& ep : execution_providers_) {
        if (ep->Name() == "CUDA") {
            ORT_RETURN_IF_ERROR(
                RegisterCudaContribKernels(ep_kernel_registry_));
        }
        // ... other EPs
    }

    // ...
}
```

---

## 48.7 How to Add New Contrib Ops

### 48.7.1 Step-by-Step Guide

#### Step 1: Define the Operator Schema

```cpp
// contrib_ops/cpu/my_new_op.h
#pragma once
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class MyNewOp : public OpKernel {
public:
    static std::unique_ptr<KernelCreateInfo> GetKernelCreateInfo();

    explicit MyNewOp(const OpKernelInfo& info);
    Status Compute(OpKernelContext* context) const override;

private:
    // Attributes
    int64_t my_attribute_;
};

}  // namespace contrib
}  // namespace onnxruntime
```

#### Step 2: Implement the Operator

```cpp
// contrib_ops/cpu/my_new_op.cc
#include "contrib_ops/cpu/my_new_op.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace contrib {

MyNewOp::MyNewOp(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("my_attribute", &my_attribute_).IsOK(),
                "Failed to get my_attribute");
}

Status MyNewOp::Compute(OpKernelContext* context) const {
    // Get inputs
    const Tensor* input = context->Input<Tensor>(0);
    ORT_RETURN_IF_NOT(input != nullptr, "Input tensor is null");

    auto input_shape = input->Shape();
    auto input_data = input->Data<float>();

    // Compute output shape
    TensorShape output_shape = input_shape;

    // Allocate output
    Tensor* output = context->Output(0, output_shape);
    auto output_data = output->MutableData<float>();

    // Parallel computation using thread pool
    auto* tp = context->GetOperatorThreadPool();
    int64_t total_elements = input_shape.Size();

    ThreadPool::ParallelFor(tp, total_elements, /*cost_per_unit=*/1.0,
        [input_data, output_data, this](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                output_data[i] = ComputeElement(input_data[i], my_attribute_);
            }
        });

    return Status::OK();
}

// Kernel creation info
std::unique_ptr<KernelCreateInfo> MyNewOp::GetKernelCreateInfo() {
    auto kernel_def = KernelDefBuilder()
        .SetName("MyNewOp")
        .SetDomain("com.microsoft")
        .SinceVersion(1)
        .Provider(kCpuExecutionProvider)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .Build();

    return std::make_unique<KernelCreateInfo>(
        std::move(kernel_def),
        [](const OpKernelInfo& info) -> std::unique_ptr<OpKernel> {
            return std::make_unique<MyNewOp>(info);
        });
}

// Registration macro (alternative approach)
ONNX_OPERATOR_KERNEL_EX(
    MyNewOp,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MyNewOp);

}  // namespace contrib
}  // namespace onnxruntime
```

#### Step 3: Register the Operator

```cpp
// contrib_ops/cpu/cpu_contrib_kernels.cc
// Add to the registration list:
static const BuildKernelCreateInfoFn kernel_create_fn_list[] = {
    // ... existing registrations ...
    BuildKernelCreateInfo<onnxruntime::contrib::MyNewOp>,  // ADD THIS
};
```

#### Step 4: Add to Build System

```cmake
# contrib_ops/cpu/CMakeLists.txt
set(CONTRIB_CPU_OPS
    # ... existing files ...
    my_new_op.cc                    # ADD THIS
)
```

#### Step 5: Add the Op to the ONNX Schema Registry (Optional)

```python
# If the op needs to be recognized by ONNX model checker:
# onnxruntime/core/graph/contrib_ops/
# Create a schema definition file

import onnx
from onnx import defs, helper

# Define the op schema
schema = defs.OpSchema(
    "MyNewOp",
    "com.microsoft",
    since_version=1,
    doc="My new contrib operator."
)
schema.Input(0, "input", "Input tensor", "T")
schema.Output(0, "output", "Output tensor", "T")
schema.Attr("my_attribute", "An attribute", AttrType.INTS, default_value=[1])
schema.TypeConstraint("T", ["tensor(float)"], "Constrain to float tensors")

# Register
defs.onnx_opset_version = 1
```

#### Step 6: Write Tests

```cpp
// contrib_ops/cpu/test/test_my_new_op.cc
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(MyNewOpTest, BasicTest) {
    OpTester tester("MyNewOp", 1, "com.microsoft");
    tester.AddAttribute("my_attribute", int64_t(42));

    std::vector<int64_t> input_shape = {2, 3};
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    tester.AddInput<float>("input", input_shape, input_data);
    tester.AddOutput<float>("output", input_shape,
                            ExpectedOutput(input_data, 42));

    tester.Run();
}

TEST(MyNewOpTest, DynamicShape) {
    OpTester tester("MyNewOp", 1, "com.microsoft");
    tester.AddAttribute("my_attribute", int64_t(10));

    // Test with dynamic dimensions
    tester.AddInput<float>("input", {2, 3}, {1, 2, 3, 4, 5, 6});
    tester.AddOutput<float>("output", {2, 3}, {/* expected */});

    tester.Run(OpTester::ExpectResult::kExpectSuccess,
               "", {kCpuExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
```

---

## 48.8 Complete Contrib Op Catalog

### 48.8.1 All CPU Contrib Operators

| Operator | Domain | Description |
|----------|--------|-------------|
| Attention | com.microsoft | Fused self-attention |
| BiasGelu | com.microsoft | Bias + GELU fusion |
| BiasDropout | com.microsoft | Bias + Dropout fusion |
| BiasSplitGelu | com.microsoft | Bias + Split + GELU |
| BeamSearch | com.microsoft | Beam search decoding |
| CropAndResize | com.microsoft | Crop and resize images |
| DynamicQuantizeMatMul | com.microsoft | Dynamic quantized MatMul |
| EmbedLayerNormalization | com.microsoft | Embedding + LayerNorm fusion |
| ExpandDims | com.microsoft | Expand tensor dimensions |
| FastGelu | com.microsoft | Fast GELU approximation |
| FusedConv | com.microsoft | Fused convolution |
| FusedGemm | com.microsoft | Fused GEMM with activation |
| FusedMatMul | com.microsoft | Fused MatMul with activation |
| Gelu | com.microsoft | GELU activation |
| GridSample | com.microsoft | Grid sampling |
| Inverse | com.microsoft | Matrix inverse |
| LayerNorm | com.microsoft | Layer normalization |
| LongformerAttention | com.microsoft | Longformer attention |
| MatMulNBits | com.microsoft | N-bit quantized MatMul |
| MatMulIntegerToFloat | com.microsoft | Int MatMul + dequantize |
| MatMulSparse | com.microsoft | Sparse MatMul |
| MultiHeadAttention | com.microsoft | Multi-head attention |
| MurmurHash3 | com.microsoft | MurmurHash3 |
| NchwcConv | com.microsoft | NCHWc Conv |
| NchwcPool | com.microsoft | NCHWc Pool |
| NchwcReorder | com.microsoft | NCHW↔NCHWc conversion |
| NvEmbedding | com.microsoft | NV Embedding lookup |
| QLinearConcat | com.microsoft | Quantized concat |
| QLinearGlobalAveragePool | com.microsoft | Quantized global avg pool |
| QLinearLookupTable | com.microsoft | Quantized lookup table |
| QLinearPool | com.microsoft | Quantized pool |
| Range | com.microsoft | Range generation |
| RegexSplit | com.microsoft | Regex text splitting |
| RegexSplitWithOffsets | com.microsoft | Regex split with offsets |
| RelativeAttentionBias | com.microsoft | Relative attention bias |
| SampleOp | com.microsoft | Sampling operation |
| SkipLayerNormalization | com.microsoft | Skip + LayerNorm fusion |
| SkipSimplifiedLayerNorm | com.microsoft | Skip + simplified LayerNorm |
| Tokenizer | com.microsoft | Text tokenization |
| TorchEmbedding | com.microsoft | Torch-style embedding |
| Trilu | com.microsoft | Triangular matrix |
| Unique | com.microsoft | Unique elements |
| WordEmbedding | com.microsoft | Word embedding lookup |

### 48.8.2 All CUDA Contrib Operators

| Operator | Domain | Description |
|----------|--------|-------------|
| Attention | com.microsoft | GPU fused attention (Flash Attention) |
| BiasGelu | com.microsoft | GPU Bias + GELU |
| BiasSplitGelu | com.microsoft | GPU Bias + Split + GELU |
| EmbedLayerNormalization | com.microsoft | GPU Embed + LayerNorm |
| FastGelu | com.microsoft | GPU Fast GELU |
| FusedConv | com.microsoft | GPU fused Conv |
| Gelu | com.microsoft | GPU GELU |
| LayerNorm | com.microsoft | GPU LayerNorm |
| LongformerAttention | com.microsoft | GPU Longformer attention |
| MatMulNBits | com.microsoft | GPU N-bit quantized MatMul |
| MatMulIntegerToFloat | com.microsoft | GPU int MatMul + dequantize |
| MultiHeadAttention | com.microsoft | GPU multi-head attention |
| QLinearConcat | com.microsoft | GPU quantized concat |
| SkipLayerNormalization | com.microsoft | GPU skip + LayerNorm |
| SkipSimplifiedLayerNorm | com.microsoft | GPU skip + simplified LayerNorm |

### 48.8.3 All WebGPU Contrib Operators

| Operator | Domain | Description |
|----------|--------|-------------|
| Attention | com.microsoft | WebGPU attention |
| EmbedLayerNormalization | com.microsoft | WebGPU embed + LayerNorm |
| SkipLayerNormalization | com.microsoft | WebGPU skip + LayerNorm |
| FastGelu | com.microsoft | WebGPU Fast GELU |
| BiasGelu | com.microsoft | WebGPU Bias + GELU |
| MatMulNBits | com.microsoft | WebGPU N-bit MatMul |

---

## 48.9 Summary

| Topic | Key Points |
|-------|-----------|
| Contrib Ops | Non-standard ONNX ops optimized for specific targets |
| Domain | `com.microsoft` for all Microsoft contrib ops |
| CPU Ops | 40+ operators including Attention, MatMulNBits, LayerNorm, etc. |
| CUDA Ops | GPU-optimized versions with Flash Attention, Tensor Core support |
| WebGPU Ops | WGSL compute shader implementations |
| Registration | `cpu_contrib_kernels.cc` / `cuda_contrib_kernels.cc` |
| Adding New Ops | Define class, implement Compute(), register, add to build |
