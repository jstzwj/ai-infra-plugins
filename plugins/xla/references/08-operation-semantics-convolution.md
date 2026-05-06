# XLA Operation Semantics: Convolution, FFT, and Linear Solve Operations

This reference provides comprehensive documentation of XLA convolution operations, Fast Fourier Transform operations, and triangular solve operations. Convolution operations are among the most complex in XLA, supporting a wide range of configurations for deep learning, signal processing, and scientific computing workloads.

---

## Table of Contents

1. [Conv (Basic)](#conv-basic)
2. [ConvWithGeneralPadding](#convwithgeneralpadding)
3. [ConvWithGeneralDimensions](#convwithgeneraldimensions)
4. [ConvGeneral](#convgeneral)
5. [ConvGeneralDilated](#convgeneraldilated)
6. [FFT](#fft)
7. [TriangularSolve](#triangularsolve)
8. [StableHLO Cross-References](#stablehlo-cross-references)

---

## Conv (Basic)

`Conv` (also called `Convolution`) computes a convolution of two tensors (input and kernel) with optional window stride and padding. This is the simplest convolution interface, suitable for standard deep learning convolutions.

### Signature

```
Conv(lhs, rhs, window_strides, feature_group_count)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `lhs` | `XlaOp` | The input (activation) tensor. Shape: `[batch, spatial_dims..., features]`. |
| `rhs` | `XlaOp` | The kernel (weight) tensor. Shape: `[spatial_dims..., input_features, output_features]`. |
| `window_strides` | `std::vector<int64>` | Stride for each spatial dimension. Controls the spacing between consecutive windows. |
| `feature_group_count` | `int64` | Number of feature groups for grouped convolution. Default 1. |

### Dimension Descriptions

XLA follows a specific dimension ordering convention for convolution operands:

**Input (lhs)**: `[batch, spatial_dim_0, spatial_dim_1, ..., spatial_dim_N, features]`
- `batch`: Number of examples in the batch.
- `spatial_dim_i`: Spatial extent of the input (e.g., height, width for 2D convolution).
- `features`: Number of input feature channels.

**Kernel (rhs)**: `[spatial_dim_0, spatial_dim_1, ..., spatial_dim_N, input_features, output_features]`
- `spatial_dim_i`: Spatial extent of the kernel (e.g., 3x3 kernel).
- `input_features`: Must equal `lhs.features / feature_group_count`.
- `output_features`: Number of output feature channels.

### Example: 2D Convolution

Input: `f32[1, 8, 8, 3]` (batch=1, height=8, width=8, channels=3)
Kernel: `f32[3, 3, 3, 64]` (kernel_h=3, kernel_w=3, in_channels=3, out_channels=64)

```
%result = f32[1, 6, 6, 64] convolution(
  f32[1, 8, 8, 3] %input,
  f32[3, 3, 3, 64] %kernel
), window_strides={1, 1}, feature_group_count=1
```

Output shape: `[batch, out_h, out_w, out_channels]` = `[1, 6, 6, 64]`

### HLO Text Format

```
%result = f32[1,6,6,64]{3,2,1,0} convolution(
  f32[1,8,8,3]{3,2,1,0} %input,
  f32[3,3,3,64]{3,2,1,0} %kernel
), window_strides={1,1}, padding={{0,0},{0,0}},
  feature_group_count=1
```

---

## ConvWithGeneralPadding

`ConvWithGeneralPadding` extends `Conv` by allowing explicit padding specification for each spatial dimension.

### Signature

```
ConvWithGeneralPadding(lhs, rhs, window_strides, padding,
                        feature_group_count)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `lhs` | `XlaOp` | Input tensor. |
| `rhs` | `XlaOp` | Kernel tensor. |
| `window_strides` | `std::vector<int64>` | Stride for each spatial dimension. |
| `padding` | `std::vector<std::pair<int64, int64>>` | Explicit padding for each spatial dimension: `{(low_0, high_0), (low_1, high_1), ...}`. Negative padding means trimming. |
| `feature_group_count` | `int64` | Number of feature groups. Default 1. |

### Padding Semantics

Each spatial dimension can have independent low and high padding:
- `low` padding: Number of elements to add before the first element.
- `high` padding: Number of elements to add after the last element.

Padding values are added as zeros (for floating point types).

### Example: Same Padding

For a 3x3 kernel with stride 1 on an 8x8 input, "same" padding is `{(1, 1), (1, 1)}`:

```
%result = f32[1,8,8,64] convolution(
  f32[1,8,8,3] %input,
  f32[3,3,3,64] %kernel
), window_strides={1,1}, padding={{1,1},{1,1}},
  feature_group_count=1
```

Output: `[1, 8, 8, 64]` (same spatial size as input due to padding).

---

## ConvWithGeneralDimensions

`ConvWithGeneralDimensions` extends `Conv` by allowing custom dimension numbering for both input and kernel tensors.

### Signature

```
ConvWithGeneralDimensions(lhs, rhs, window_strides, padding,
                          dimension_numbers, feature_group_count)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `lhs` | `XlaOp` | Input tensor. |
| `rhs` | `XlaOp` | Kernel tensor. |
| `window_strides` | `std::vector<int64>` | Stride for each spatial dimension. |
| `padding` | `std::vector<std::pair<int64, int64>>` | Padding for each spatial dimension. |
| `dimension_numbers` | `ConvolutionDimensionNumbers` | Custom dimension numbering. |
| `feature_group_count` | `int64` | Number of feature groups. |

### ConvolutionDimensionNumbers

This struct maps abstract dimension roles to concrete dimension indices:

```cpp
struct ConvolutionDimensionNumbers {
  int64 input_batch_dimension;        // e.g., 0 for NCHW
  int64 input_feature_dimension;      // e.g., 1 for NCHW

  std::vector<int64> input_spatial_dimensions;  // e.g., {2, 3} for NCHW

  int64 kernel_input_feature_dimension;   // e.g., 1 for OIHW
  int64 kernel_output_feature_dimension;  // e.g., 0 for OIHW

  std::vector<int64> kernel_spatial_dimensions;  // e.g., {2, 3} for OIHW

  int64 output_batch_dimension;        // e.g., 0 for NCHW
  int64 output_feature_dimension;      // e.g., 1 for NCHW

  std::vector<int64> output_spatial_dimensions;  // e.g., {2, 3} for NCHW
};
```

### Example: NCHW Format

```
// Input: NCHW = [batch, channels, height, width]
// Kernel: OIHW = [out_channels, in_channels, kernel_h, kernel_w]

dimension_numbers = ConvolutionDimensionNumbers(
  input_batch_dimension = 0,
  input_feature_dimension = 1,
  input_spatial_dimensions = {2, 3},
  kernel_output_feature_dimension = 0,
  kernel_input_feature_dimension = 1,
  kernel_spatial_dimensions = {2, 3},
  output_batch_dimension = 0,
  output_feature_dimension = 1,
  output_spatial_dimensions = {2, 3}
)
```

With this configuration, an NCHW input `[1, 3, 8, 8]` and OIHW kernel `[64, 3, 3, 3]` produces output `[1, 64, 6, 6]`.

### Example: NHWC Format (Default)

```
dimension_numbers = ConvolutionDimensionNumbers(
  input_batch_dimension = 0,
  input_feature_dimension = 3,
  input_spatial_dimensions = {1, 2},
  kernel_output_feature_dimension = 3,
  kernel_input_feature_dimension = 2,
  kernel_spatial_dimensions = {0, 1},
  output_batch_dimension = 0,
  output_feature_dimension = 3,
  output_spatial_dimensions = {1, 2}
)
```

---

## ConvGeneral

`ConvGeneral` extends `ConvWithGeneralDimensions` by also supporting batch grouping (used for channelwise/depthwise-separable convolutions in certain formulations).

### Signature

```
ConvGeneral(lhs, rhs, window_strides, padding, dimension_numbers,
            feature_group_count, batch_group_count)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `lhs` | `XlaOp` | Input tensor. |
| `rhs` | `XlaOp` | Kernel tensor. |
| `window_strides` | `std::vector<int64>` | Stride for each spatial dimension. |
| `padding` | `std::vector<std::pair<int64, int64>>` | Padding. |
| `dimension_numbers` | `ConvolutionDimensionNumbers` | Dimension mapping. |
| `feature_group_count` | `int64` | Number of feature groups (grouped convolution). |
| `batch_group_count` | `int64` | Number of batch groups. Default 1. |

### Batch Groups

`batch_group_count` divides the batch dimension into groups. For each group, a separate convolution is computed. This is an alternative way to implement grouped or depthwise convolutions.

When `batch_group_count > 1`:
- `lhs.batch` must be divisible by `batch_group_count`.
- `rhs.output_features` must be divisible by `batch_group_count`.
- The batch dimension of the input is split into `batch_group_count` groups, and each group is convolved with the corresponding slice of the kernel.

---

## ConvGeneralDilated

`ConvGeneralDilated` is the most general convolution operation in XLA, supporting dilation (both on the input and the kernel), window reversal, and precision configuration.

### Signature

```
ConvGeneralDilated(lhs, rhs, window_strides, padding, lhs_dilation,
                   rhs_dilation, dimension_numbers, feature_group_count,
                   batch_group_count, window_reversal, precision_config,
                   preferred_element_type)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `lhs` | `XlaOp` | Input tensor (activation). |
| `rhs` | `XlaOp` | Kernel tensor (weights). |
| `window_strides` | `std::vector<int64>` | Stride for each spatial dimension. |
| `padding` | `std::vector<std::pair<int64, int64>>` | Explicit low/high padding per spatial dimension. |
| `lhs_dilation` | `std::vector<int64>` | Dilation (upsampling) factor applied to the input (`lhs`) spatial dimensions. Also known as **transposed convolution** dilation. |
| `rhs_dilation` | `std::vector<int64>` | Dilation factor applied to the kernel (`rhs`) spatial dimensions. Also known as **atrous** or **dilated convolution**. |
| `dimension_numbers` | `ConvolutionDimensionNumbers` | Dimension mapping between operands. |
| `feature_group_count` | `int64` | Feature group count for grouped convolution. |
| `batch_group_count` | `int64` | Batch group count. Default 1. |
| `window_reversal` | `std::vector<bool>` | Whether to reverse the kernel along each spatial dimension. Useful for transposed convolution. |
| `precision_config` | `std::vector<PrecisionConfig::Precision>` | Precision for each operand. |
| `preferred_element_type` | `std::optional<PrimitiveType>` | The preferred element type for the output. Allows specifying a different accumulation type. |

### Dilation Semantics

#### Atrous (Dilated) Convolution: `rhs_dilation`

When `rhs_dilation[i] > 1`, the kernel is effectively expanded by inserting zeros between its elements along dimension `i`. This increases the receptive field without increasing the number of parameters.

For a kernel element at spatial position `(k_0, k_1, ...)`:
```
effective_position[i] = k_i * rhs_dilation[i]
```

This is commonly used in semantic segmentation (e.g., DeepLab) to increase the field of view while maintaining spatial resolution.

**Example**: 3x3 kernel with `rhs_dilation = {2, 2}` effectively becomes a 5x5 kernel (with holes):
```
Original:         Dilated:
[x x x]          [x . x . x]
[x x x]          [. . . . .]
[x x x]          [x . x . x]
                  [. . . . .]
                  [x . x . x]
```

#### Transposed Convolution: `lhs_dilation`

When `lhs_dilation[i] > 1`, zeros are inserted between input elements along dimension `i`. Combined with appropriate padding, this effectively performs an upsampling (transposed/deconvolution) operation.

For an input element at spatial position `(p_0, p_1, ...)`:
```
effective_position[i] = p_i * lhs_dilation[i]
```

**Example**: 2x2 input with `lhs_dilation = {2, 2}`:
```
Original:    Dilated:
[a b]        [a . b .]
[c d]        [. . . .]
             [c . d .]
             [. . . .]
```

### Feature Groups and Batch Groups

#### Grouped Convolution (`feature_group_count > 1`)

The input features are divided into `feature_group_count` groups. Each group is convolved independently with its own set of filters.

```
input_features_per_group = input_features / feature_group_count
output_features_per_group = output_features / feature_group_count
```

The kernel shape constraint: `kernel.input_features = input_features / feature_group_count`

#### Depthwise Convolution

Depthwise convolution is a special case of grouped convolution where `feature_group_count = input_features` and `kernel.output_features = input_features * channel_multiplier` (typically `channel_multiplier = 1`, so `kernel.output_features = input_features`).

```
feature_group_count = input_channels
kernel: [kH, kW, input_channels, channel_multiplier]
```

Each input channel is convolved with its own filter independently.

### Window Reversal

The `window_reversal` parameter reverses the kernel along specified spatial dimensions before convolution. When `window_reversal[i] = true`, dimension `i` of the kernel is flipped.

This is used in transposed convolution implementations where the kernel must be flipped to compute the transpose of the forward convolution.

### Precision Config

The `precision_config` specifies the desired precision for each operand during computation:

| Precision Value | Description |
|---|---|
| `DEFAULT` | Backend's default precision (typically highest available). |
| `HIGH` | High precision (e.g., FP32 on GPU tensor cores). |
| `HIGHEST` | Highest available precision. |

On GPUs with tensor cores, the precision config controls whether tensor cores are used:
- `HIGH` may allow mixed-precision tensor core operations (FP16 inputs, FP32 accumulation).
- `HIGHEST` forces full-precision computation.

### Preferred Element Type

The `preferred_element_type` allows specifying the output element type, which may differ from the input types. This is useful for:
- Performing FP16 convolution with FP32 output accumulation.
- Performing BF16 convolution with FP32 output.
- Performing INT8 convolution with INT32 accumulation.

### Output Shape Calculation

For a convolution with the following parameters:
- Input spatial size: `I`
- Kernel spatial size: `K`
- Padding: `(P_low, P_high)`
- Window stride: `S`
- Input (lhs) dilation: `D_lhs`
- Kernel (rhs) dilation: `D_rhs`

The output spatial size for each dimension:
```
padded_input = P_low + (I - 1) * D_lhs + 1 + P_high
effective_kernel = (K - 1) * D_rhs + 1
output = (padded_input - effective_kernel) / S + 1
```

### Pseudo-Code

The following pseudo-code illustrates the complete convolution computation:

```
// For each output element
for b in range(batch_size):
  for f_out in range(output_features):
    for spatial_out in output_spatial_range:
      accumulator = 0

      // Sum over the kernel window
      for f_group_idx in range(feature_group_count):
        f_in_start = f_group_idx * (input_features / feature_group_count)
        f_in_end = f_in_start + (input_features / feature_group_count)

        for f_in in range(f_in_start, f_in_end):
          for kernel_spatial in kernel_spatial_range:
            // Apply rhs dilation to kernel position
            k_pos = kernel_spatial * rhs_dilation

            // Compute input position
            input_pos = spatial_out * window_stride + k_pos

            // Apply lhs dilation
            input_pos_dilated = input_pos  // after accounting for lhs_dilation

            // Account for padding
            input_pos_padded = input_pos_dilated - padding_low

            // Bounds check (skip out-of-bounds positions)
            if all(0 <= input_pos_padded[i] < input_spatial[i]):
              // Apply window reversal
              k_idx = kernel_spatial
              if window_reversal[dim]:
                k_idx = reverse(kernel_spatial, kernel_size)

              // Accumulate
              accumulator += input[b, input_pos_padded, f_in] *
                             kernel[k_idx, f_in % group_size, f_out]

      output[b, spatial_out, f_out] = accumulator
```

### Example: Standard 2D Convolution

```
// Input: f32[2, 8, 8, 3]  (NHWC)
// Kernel: f32[3, 3, 3, 64] (HWIO)
// Stride: {1, 1}
// Padding: {{0, 0}, {0, 0}} (valid)

%conv = f32[2, 6, 6, 64] convolution(
  f32[2, 8, 8, 3] %input,
  f32[3, 3, 3, 64] %kernel
), window_strides={1, 1}, padding={{0,0},{0,0}},
  dimension_numbers=NHWC_HWIO,
  feature_group_count=1, batch_group_count=1,
  lhs_dilation={1, 1}, rhs_dilation={1, 1},
  window_reversal={0, 0}
```

### Example: Dilated (Atrous) Convolution

```
// Input: f32[1, 8, 8, 3]
// Kernel: f32[3, 3, 3, 64]
// rhs_dilation = {2, 2} -> effective kernel size 5x5

%conv = f32[1, 4, 4, 64] convolution(
  f32[1, 8, 8, 3] %input,
  f32[3, 3, 3, 64] %kernel
), window_strides={1, 1}, padding={{0,0},{0,0}},
  rhs_dilation={2, 2}
// Output: 8 - 5 + 1 = 4 per spatial dim
```

### Example: Transposed Convolution

```
// Input: f32[1, 4, 4, 3]
// Kernel: f32[3, 3, 3, 64]
// lhs_dilation = {2, 2}, padding = {{2, 2}, {2, 2}}
// This upsamples 4x4 -> 8x8

%conv = f32[1, 8, 8, 64] convolution(
  f32[1, 4, 4, 3] %input,
  f32[3, 3, 3, 64] %kernel
), window_strides={1, 1},
  padding={{2,2},{2,2}},
  lhs_dilation={2, 2},
  window_reversal={1, 1}
```

### Example: Depthwise Convolution

```
// Input: f32[1, 8, 8, 3]
// Kernel: f32[3, 3, 3, 1] (3 input channels, channel_multiplier=1)
// feature_group_count = 3 (one group per channel)

%conv = f32[1, 6, 6, 3] convolution(
  f32[1, 8, 8, 3] %input,
  f32[3, 3, 3, 1] %kernel
), window_strides={1, 1}, padding={{0,0},{0,0}},
  feature_group_count=3
```

### Example: Grouped Convolution

```
// Input: f32[1, 8, 8, 128]
// Kernel: f32[3, 3, 32, 128]  (32 = 128/4 input features per group)
// feature_group_count = 4

%conv = f32[1, 6, 6, 128] convolution(
  f32[1, 8, 8, 128] %input,
  f32[3, 3, 32, 128] %kernel
), window_strides={1, 1}, padding={{0,0},{0,0}},
  feature_group_count=4
```

---

## FFT

`FFT` computes a Fast Fourier Transform on the innermost dimensions of the input tensor. XLA supports forward and inverse FFT operations, including complex-to-complex, real-to-complex, and complex-to-real transforms.

### FftType Enum

| Value | Transform | Input Type | Output Type | Description |
|---|---|---|---|---|
| `FFT` | Forward complex FFT | Complex | Complex | Standard forward DFT |
| `IFFT` | Inverse complex FFT | Complex | Complex | Standard inverse DFT |
| `RFFT` | Forward real FFT | Real (float) | Complex | Forward DFT of real input; output is conjugate-symmetric, so only the first `N/2 + 1` values are computed |
| `IRFFT` | Inverse real FFT | Complex | Real (float) | Inverse DFT assuming conjugate-symmetric input; output is real-valued |

### Signature

```
FFT(operand, fft_type, fft_length)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `operand` | `XlaOp` | The input tensor. For `FFT`/`IFFT`, must be of complex type (`C64` or `C128`). For `RFFT`, must be of floating point type (`F32` or `F64`). For `IRFFT`, must be of complex type. |
| `fft_type` | `FftType` | The type of FFT to compute: `FFT`, `IFFT`, `RFFT`, or `IRFFT`. |
| `fft_length` | `std::vector<int64>` | The length of the FFT along each transformed dimension. If the input is shorter than `fft_length`, it is zero-padded. If longer, it is truncated. |

### Semantics

The FFT is applied to the innermost `len(fft_length)` dimensions of the operand. The outer dimensions are treated as batch dimensions and are not transformed.

#### Output Shape

| Transform | Input Shape | fft_length | Output Shape |
|---|---|---|---|
| `FFT` | `[..., N]` | `[N]` | `[..., N]` (complex) |
| `IFFT` | `[..., N]` | `[N]` | `[..., N]` (complex) |
| `RFFT` | `[..., N]` | `[N]` | `[..., N/2 + 1]` (complex) |
| `IRFFT` | `[..., N]` | `[N]` | `[..., N]` (real) |

For multi-dimensional FFTs:
- `fft_length = [M, N]` applies a 2D FFT on the last two dimensions.
- The output shape changes only for `RFFT` on the innermost transformed dimension.

### Example: 1D Forward FFT

Input: `f32[8]` (real values)
Transform: `RFFT`, `fft_length = [8]`

```
%result = c64[5] fft(f32[8] %input), fft_type=RFFT, fft_length={8}
```

Output: `c64[5]` (first 5 = 8/2 + 1 complex frequency bins)

### Example: 1D Inverse FFT

Input: `c64[8]` (complex values)
Transform: `IFFT`, `fft_length = [8]`

```
%result = c64[8] fft(c64[8] %input), fft_type=IFFT, fft_length={8}
```

### Example: 2D FFT

Input: `f32[1, 16, 16]` (batch of 1, spatial 16x16)
Transform: `RFFT`, `fft_length = [16, 16]`

```
%result = c64[1, 16, 9] fft(f32[1, 16, 16] %input),
  fft_type=RFFT, fft_length={16, 16}
```

Output: `c64[1, 16, 9]` -- the last dimension is `16/2 + 1 = 9`.

### Example: Complex-to-Complex 1D FFT

Input: `c64[4, 32]` (batch of 4, length 32)
Transform: `FFT`, `fft_length = [32]`

```
%result = c64[4, 32] fft(c64[4, 32] %input),
  fft_type=FFT, fft_length={32}
```

### HLO Text Format

```
%result = c64[5]{0} fft(f32[8]{0} %input),
  fft_type=RFFT, fft_length={8}

%result = c64[4,32]{1,0} fft(c64[4,32]{1,0} %input),
  fft_type=FFT, fft_length={32}
```

### Normalization

XLA FFT operations follow standard signal processing conventions:

- **FFT (forward)**: No normalization (sum of products).
  ```
  X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*k*n/N)
  ```

- **IFFT (inverse)**: Normalized by `1/N`.
  ```
  x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(2*pi*i*k*n/N)
  ```

- **RFFT**: Same as FFT but exploits real-valued input for efficiency.
- **IRFFT**: Same as IFFT but returns only the real part, normalized by `1/N`.

Note: The normalization convention may differ from some libraries (e.g., NumPy's `fft` does not normalize, and `ifft` normalizes by `1/N`).

### Efficiency Considerations

- FFT lengths that are powers of 2 are most efficient.
- XLA may use mixed-radix algorithms for non-power-of-2 lengths, but performance may be lower.
- The batch dimensions are processed in parallel, so batching is encouraged for throughput.

---

## TriangularSolve

`TriangularSolve` solves a system of linear equations with a triangular coefficient matrix. Given a lower or upper triangular matrix `a` and a right-hand side `b`, it computes `x` such that `a * x = b` (or `x * a = b` for right-side solves).

### Signature

```
TriangularSolve(a, b, left_side, lower, unit_diagonal,
                transpose_a, convergence_limit, max_iter,
                precision_config)
```

### Arguments

| Argument | Type | Description |
|---|---|---|
| `a` | `XlaOp` | The triangular coefficient matrix. Shape: `[..., M, M]`. Only the lower or upper triangle is used, depending on the `lower` flag. |
| `b` | `XlaOp` | The right-hand side. Shape: `[..., M, K]` if `left_side=true`, or `[..., K, M]` if `left_side=false`. |
| `left_side` | `bool` | If `true`, solve `a * x = b`. If `false`, solve `x * a = b`. Default `true`. |
| `lower` | `bool` | If `true`, `a` is lower triangular (only the lower triangle is read). If `false`, `a` is upper triangular. Default `true`. |
| `unit_diagonal` | `bool` | If `true`, the diagonal of `a` is assumed to be all ones (not read from `a`). Useful for Cholesky-based solves where the diagonal is known to be unit. Default `false`. |
| `transpose_a` | `Transpose` | Whether and how to transpose `a` before solving. Options: `NO_TRANSPOSE`, `TRANSPOSE`, `ADJOINT` (conjugate transpose). Default `NO_TRANSPOSE`. |
| `convergence_limit` | `std::optional<float>` | Not currently used. |
| `max_iter` | `std::optional<int64>` | Not currently used. |
| `precision_config` | `std::vector<PrecisionConfig::Precision>` | Precision for each operand. |

### Semantics

Solves the triangular system of equations. The leading dimensions of `a` and `b` are batch dimensions, and the solve is applied independently to each batch element.

**When `left_side = true` and `transpose_a = NO_TRANSPOSE`:**
```
a * x = b  =>  x = a^(-1) * b
```

**When `left_side = false` and `transpose_a = NO_TRANSPOSE`:**
```
x * a = b  =>  x = b * a^(-1)
```

**When `transpose_a = TRANSPOSE`:**
```
a^T * x = b  (left_side)  or  x * a^T = b  (right_side)
```

**When `transpose_a = ADJOINT`:**
```
a^H * x = b  (left_side)  or  x * a^H = b  (right_side)
```

### Output Shape

Same shape as `b`.

### Example: Simple 3x3 Lower Triangular Solve

```
a = [[2, 0, 0],
     [1, 3, 0],
     [4, 2, 1]]

b = [[4],
     [7],
     [8]]

// Solve a * x = b
// 2*x0 = 4 => x0 = 2
// 1*x0 + 3*x1 = 7 => x1 = 5/3
// 4*x0 + 2*x1 + 1*x2 = 8 => x2 = 8 - 8 - 10/3 = -14/3
```

```
%result = f32[3, 1] triangular-solve(
  f32[3, 3] %a,
  f32[3, 1] %b
), left_side=true, lower=true, unit_diagonal=false,
  transpose_a=NO_TRANSPOSE
```

### Example: Batched Solve

```
// a: f32[4, 3, 3]  (batch of 4, each 3x3 lower triangular)
// b: f32[4, 3, 2]  (batch of 4, each with 2 right-hand sides)
// Solve a[i] * x[i] = b[i] for each i in 0..3

%result = f32[4, 3, 2] triangular-solve(
  f32[4, 3, 3] %a,
  f32[4, 3, 2] %b
), left_side=true, lower=true
```

### Example: Upper Triangular with Transpose

```
// a: f32[4, 4] upper triangular
// b: f32[4] right-hand side
// Solve a^T * x = b

%result = f32[4] triangular-solve(
  f32[4, 4] %a,
  f32[4] %b
), left_side=true, lower=false, transpose_a=TRANSPOSE
```

### Example: Right-Side Solve

```
// a: f32[3, 3] lower triangular
// b: f32[5, 3] (K=5 right-hand sides, each of size 3)
// Solve x * a = b  =>  x = b * a^(-1)

%result = f32[5, 3] triangular-solve(
  f32[3, 3] %a,
  f32[5, 3] %b
), left_side=false, lower=true
```

### HLO Text Format

```
%result = f32[3,1]{1,0} triangular-solve(
  f32[3,3]{1,0} %a,
  f32[3,1]{1,0} %b
), left_side=true, lower=true, unit_diagonal=false,
  transpose_a=NO_TRANSPOSE
```

### Use Cases

1. **Cholesky decomposition follow-up**: After `Cholesky(A)` produces a lower triangular `L`, solve `L * x = b` to compute `A^(-1) * b` without explicitly inverting `A`.

2. **QR decomposition follow-up**: After QR decomposition, solve `R * x = Q^T * b` where `R` is upper triangular.

3. **LU decomposition follow-up**: After LU decomposition with pivoting, perform two triangular solves: `L * y = P * b` (forward substitution), then `U * x = y` (back substitution).

---

## StableHLO Cross-References

| XLA Operation | StableHLO Operation | Notes |
|---|---|---|
| ConvGeneralDilated | `stablehlo.convolution` | Unified convolution operation with all parameters |
| FFT | `stablehlo.fft` | Same FFT types and semantics |
| TriangularSolve | `stablehlo.triangular_solve` | Same semantics |

### StableHLO Example: Convolution

```mlir
%result = stablehlo.convolution(%input, %kernel) {
  dim_numbers = #stablehlo.conv<
    input_batch_dimension = 0,
    input_feature_dimension = 3,
    input_spatial_dimensions = [1, 2],
    kernel_output_feature_dimension = 3,
    kernel_input_feature_dimension = 2,
    kernel_spatial_dimensions = [0, 1],
    output_batch_dimension = 0,
    output_feature_dimension = 3,
    output_spatial_dimensions = [1, 2]
  >,
  window_strides = dense<[1, 1]> : tensor<2xi64>,
  padding = dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>,
  lhs_dilation = dense<[1, 1]> : tensor<2xi64>,
  rhs_dilation = dense<[1, 1]> : tensor<2xi64>,
  window_reversal = dense<[false, false]> : tensor<2xi1>,
  feature_group_count = 1 : i64,
  batch_group_count = 1 : i64,
  precision_config = [DEFAULT, DEFAULT]
} : (tensor<1x8x8x3xf32>, tensor<3x3x3x64xf32>) -> tensor<1x6x6x64xf32>
```

### StableHLO Example: FFT

```mlir
%result = stablehlo.fft(%input) {
  fft_type = RFFT,
  fft_length = dense<8> : tensor<1xi64>
} : (tensor<8xf32>) -> tensor<5xcomplex<f32>>
```

### StableHLO Example: TriangularSolve

```mlir
%result = stablehlo.triangular_solve(%a, %b) {
  left_side = true,
  lower = true,
  unit_diagonal = false,
  transpose_a = NO_TRANSPOSE
} : (tensor<3x3xf32>, tensor<3x1xf32>) -> tensor<3x1xf32>
```

---

## Appendix: Convolution Variants Relationship

```
Conv
  └── adds padding ──> ConvWithGeneralPadding
       └── adds dimension_numbers ──> ConvWithGeneralDimensions
            └── adds batch_group_count ──> ConvGeneral
                 └── adds dilation, window_reversal,
                     precision_config, preferred_element_type
                     ──> ConvGeneralDilated
```

Each variant adds more configuration options, with `ConvGeneralDilated` being the most general form. In practice, the XLA builder API provides convenience methods that internally call `ConvGeneralDilated` with appropriate defaults.

### Quick Reference: Convolution Parameters

| Parameter | Default | Purpose |
|---|---|---|
| `window_strides` | All 1 | Controls output spatial resolution |
| `padding` | All (0,0) | Controls output spatial size and boundary handling |
| `lhs_dilation` | All 1 | Input dilation (transposed convolution) |
| `rhs_dilation` | All 1 | Kernel dilation (atrous/dilated convolution) |
| `feature_group_count` | 1 | Grouped convolution |
| `batch_group_count` | 1 | Alternative grouping |
| `window_reversal` | All false | Kernel flipping (transposed convolution) |
| `dimension_numbers` | NHWC/HWIO | Layout specification |
| `precision_config` | DEFAULT | Numeric precision |
| `preferred_element_type` | Same as input | Output accumulation type |
