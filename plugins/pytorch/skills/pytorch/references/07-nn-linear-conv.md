# PyTorch - Chapter 7: Linear and Convolution Layers

This reference covers all linear and convolution layers in torch.nn, including their parameters, formulas, weight shapes, and usage examples.

---

## 7.1 Linear Layers

### nn.Identity

```python
nn.Identity(*args, **kwargs)
```
Placeholder identity operator. Returns input unchanged.

### nn.Linear

```python
nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
```

Applies affine transformation: **y = xA^T + b**

- **in_features**: Size of each input sample
- **out_features**: Size of each output sample
- **bias**: If False, layer won't learn additive bias

**Weight shape**: `(out_features, in_features)`, initialized from kaiming_uniform_
**Bias shape**: `(out_features)`, initialized from uniform_(-1/sqrt(in_features), 1/sqrt(in_features))

```python
layer = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = layer(input)  # shape: (128, 30)
```

### nn.Bilinear

```python
nn.Bilinear(in1_features, in2_features, out_features, bias=True, device=None, dtype=None)
```

Applies bilinear transformation: **y = x1^T A x2 + b**

- **in1_features**: Size of first input
- **in2_features**: Size of second input
- **out_features**: Size of output

**Weight shape**: `(out_features, in1_features, in2_features)`

```python
layer = nn.Bilinear(20, 30, 40)
x1 = torch.randn(128, 20)
x2 = torch.randn(128, 30)
output = layer(x1, x2)  # shape: (128, 40)
```

### nn.LazyLinear

```python
nn.LazyLinear(out_features, bias=True, device=None, dtype=None)
```

Linear layer where `in_features` is inferred from the first input.

```python
layer = nn.LazyLinear(30)
input = torch.randn(128, 20)
output = layer(input)  # in_features inferred as 20
```

---

## 7.2 Convolution Layers

### nn.Conv1d

```python
nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0,
          dilation=1, groups=1, bias=True, padding_mode='zeros',
          device=None, dtype=None)
```

Applies 1D convolution: **y = correlation(x, weight) + bias**

**Input shape**: `(N, C_in, L_in)`
**Output shape**: `(N, C_out, L_out)` where `L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)`

**Weight shape**: `(out_channels, in_channels/groups, kernel_size)`
**Bias shape**: `(out_channels)`

```python
conv = nn.Conv1d(16, 33, 3, stride=2)
input = torch.randn(20, 16, 50)
output = conv(input)  # shape: (20, 33, 24)
```

### nn.Conv2d

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
          dilation=1, groups=1, bias=True, padding_mode='zeros',
          device=None, dtype=None)
```

**Input shape**: `(N, C_in, H_in, W_in)`
**Output shape**: `(N, C_out, H_out, W_out)`

**Weight shape**: `(out_channels, in_channels/groups, kH, kW)`

```python
# With square kernel
conv = nn.Conv2d(3, 64, 3, stride=1, padding=1)

# With non-square kernel and padding
conv = nn.Conv2d(3, 64, (3, 5), stride=(1, 2), padding=(1, 2))

input = torch.randn(16, 3, 32, 32)
output = conv(input)
```

### nn.Conv3d

```python
nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0,
          dilation=1, groups=1, bias=True, padding_mode='zeros',
          device=None, dtype=None)
```

**Input shape**: `(N, C_in, D_in, H_in, W_in)`
**Weight shape**: `(out_channels, in_channels/groups, kD, kH, kW)`

### Padding Modes

| Mode | Description |
|------|-------------|
| `'zeros'` | Pads with zeros (default) |
| `'reflect'` | Pads with reflection of input (no repeating border values) |
| `'replicate'` | Pads with replication of last pixel value |
| `'circular'` | Pads with circular wrap |

### Groups (Depthwise / Grouped Convolution)

```python
# Standard convolution: groups=1 (default)
conv = nn.Conv2d(64, 64, 3, groups=1)

# Depthwise convolution: groups=in_channels
conv = nn.Conv2d(64, 64, 3, groups=64)

# Depthwise separable = Depthwise + Pointwise
depthwise = nn.Conv2d(64, 64, 3, padding=1, groups=64)
pointwise = nn.Conv2d(64, 128, 1)

# Grouped convolution: groups=4
conv = nn.Conv2d(64, 128, 3, groups=4)
```

---

## 7.3 Transposed Convolution Layers

### nn.ConvTranspose1d

```python
nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1,
                   padding=0, output_padding=0, groups=1, bias=True,
                   dilation=1, padding_mode='zeros', device=None, dtype=None)
```

Also called "deconvolution". Computes gradient of Conv1d.

**Output length**: `L_out = (L_in - 1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1`

### nn.ConvTranspose2d

```python
nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1,
                   padding=0, output_padding=0, groups=1, bias=True,
                   dilation=1, padding_mode='zeros', device=None, dtype=None)
```

**Weight shape**: `(in_channels, out_channels/groups, kH, kW)`

```python
conv_t = nn.ConvTranspose2d(16, 33, 3, stride=2)
input = torch.randn(20, 16, 5, 5)
output = conv_t(input)  # shape: (20, 33, 11, 11)
```

### nn.ConvTranspose3d

```python
nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=1,
                   padding=0, output_padding=0, groups=1, bias=True,
                   dilation=1, padding_mode='zeros', device=None, dtype=None)
```

---

## 7.4 Lazy Convolution Variants

```python
nn.LazyConv1d(out_channels, kernel_size, ...)
nn.LazyConv2d(out_channels, kernel_size, ...)
nn.LazyConv3d(out_channels, kernel_size, ...)
nn.LazyConvTranspose1d(out_channels, kernel_size, ...)
nn.LazyConvTranspose2d(out_channels, kernel_size, ...)
nn.LazyConvTranspose3d(out_channels, kernel_size, ...)
```

`in_channels` is inferred from the first input tensor.

---

## 7.5 Unfold / Fold

```python
# Extract sliding local blocks from batched input
nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)
nn.Fold(output_size, kernel_size, dilation=1, padding=0, stride=1)
```

```python
unfold = nn.Unfold(kernel_size=(2, 3))
inp = torch.randn(2, 5, 3, 4)
output = unfold(inp)  # shape: (2, 5*2*3, 4) = (2, 30, 4)

fold = nn.Fold(output_size=(3, 4), kernel_size=(2, 3))
folded = fold(output)  # shape: (2, 5, 3, 4)
```
