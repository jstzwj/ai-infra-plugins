# PyTorch - Chapter 9: Normalization, Pooling, and Activation Layers

This reference covers all normalization, pooling, and activation layers in torch.nn.

---

## 9.1 Normalization Layers

### nn.BatchNorm1d / 2d / 3d

```python
nn.BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=True,
               track_running_stats=True, device=None, dtype=None)
```

- **num_features**: C from input (N, C) or (N, C, L)
- **eps**: Value added to variance for numerical stability
- **momentum**: Running mean/var update: running = (1-momentum)*running + momentum*batch
- **affine**: If True, has learnable weight and bias
- **track_running_stats**: If True, tracks running mean/var

**Train mode**: Uses batch statistics, updates running stats
**Eval mode**: Uses running statistics

### nn.SyncBatchNorm

```python
nn.SyncBatchNorm(num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, process_group=None, device=None, dtype=None)
```

BatchNorm synchronized across distributed processes. Convert existing BatchNorm:
```python
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
```

### nn.LayerNorm

```python
nn.LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, bias=True,
             device=None, dtype=None)
```

Normalizes over last D dimensions. `normalized_shape` can be int or list.

```python
# Normalize over last dimension
ln = nn.LayerNorm(512)
x = torch.randn(32, 10, 512)
ln(x)  # Normalizes over dim=512

# Normalize over last two dimensions
ln = nn.LayerNorm([10, 512])
```

### nn.GroupNorm

```python
nn.GroupNorm(num_groups, num_channels, eps=1e-5, affine=True,
             device=None, dtype=None)
```

Divides channels into groups and normalizes within each group. Does not depend on batch size.

```python
gn = nn.GroupNorm(32, 64)  # 32 groups, 64 channels (2 channels per group)
x = torch.randn(16, 64, 8, 8)
gn(x)
```

### nn.InstanceNorm1d / 2d / 3d

```python
nn.InstanceNorm2d(num_features, eps=1e-5, momentum=0.1, affine=False,
                  track_running_stats=False, device=None, dtype=None)
```

Normalizes each instance independently. Default: no learnable parameters (affine=False).

### nn.LocalResponseNorm

```python
nn.LocalResponseNorm(size, alpha=1e-4, beta=0.75, k=1.0)
```

---

## 9.2 Pooling Layers

### Max Pooling

```python
nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1,
             return_indices=False, ceil_mode=False)
nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1,
             return_indices=False, ceil_mode=False)
nn.MaxPool3d(kernel_size, stride=None, padding=0, dilation=1,
             return_indices=False, ceil_mode=False)
```

```python
pool = nn.MaxPool2d(2, stride=2)
x = torch.randn(1, 64, 32, 32)
pool(x)  # (1, 64, 16, 16)

# Return indices for unpooling
pool = nn.MaxPool2d(2, return_indices=True)
output, indices = pool(x)
```

### Max Unpooling

```python
nn.MaxUnpool1d(kernel_size, stride=None, padding=0)
nn.MaxUnpool2d(kernel_size, stride=None, padding=0)
nn.MaxUnpool3d(kernel_size, stride=None, padding=0)
```

```python
pool = nn.MaxPool2d(2, stride=2, return_indices=True)
unpool = nn.MaxUnpool2d(2, stride=2)
output, indices = pool(x)
reconstructed = unpool(output, indices)
```

### Average Pooling

```python
nn.AvgPool1d(kernel_size, stride=None, padding=0, ceil_mode=False,
             count_include_pad=True)
nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False,
             count_include_pad=True)
nn.AvgPool3d(kernel_size, stride=None, padding=0, ceil_mode=False,
             count_include_pad=True)
```

### Adaptive Pooling

```python
nn.AdaptiveMaxPool1d(output_size)
nn.AdaptiveMaxPool2d(output_size)
nn.AdaptiveMaxPool3d(output_size)
nn.AdaptiveAvgPool1d(output_size)
nn.AdaptiveAvgPool2d(output_size)
nn.AdaptiveAvgPool3d(output_size)
```

```python
# Global average pooling
gap = nn.AdaptiveAvgPool2d(1)  # Output: (N, C, 1, 1) regardless of input size

# Specific output size
pool = nn.AdaptiveAvgPool2d((4, 4))  # Output: (N, C, 4, 4)
```

### Other Pooling

```python
nn.LPPool1d(norm_type, kernel_size, stride=None, ceil_mode=False)
nn.LPPool2d(norm_type, kernel_size, stride=None, ceil_mode=False)
nn.FractionalMaxPool2d(kernel_size, output_size=None, output_ratio=None,
                       return_indices=False, _random_samples=None)
nn.FractionalMaxPool3d(kernel_size, output_size=None, output_ratio=None,
                       return_indices=False, _random_samples=None)
```

---

## 9.3 Activation Layers

| Layer | Formula | Parameters |
|-------|---------|------------|
| `nn.ReLU` | max(0, x) | `inplace` |
| `nn.ReLU6` | min(max(0, x), 6) | `inplace` |
| `nn.LeakyReLU` | max(0,x) + neg_slope*min(0,x) | `negative_slope=0.01, inplace` |
| `nn.PReLU` | max(0,x) + weight*min(0,x) | `num_parameters=1, init=0.25` |
| `nn.RReLU` | random leaky ReLU | `lower=1/8, upper=1/3, inplace` |
| `nn.ELU` | max(0,x) + min(0, alpha*(exp(x)-1)) | `alpha=1.0, inplace` |
| `nn.CELU` | Continuously differentiable ELU | `alpha=1.0, inplace` |
| `nn.SELU` | Scaled ELU (self-normalizing) | `inplace` |
| `nn.GELU` | x * Phi(x) | `approximate='none'` |
| `nn.SiLU` | x * sigmoid(x) (Swish) | `inplace` |
| `nn.Mish` | x * tanh(softplus(x)) | `inplace` |
| `nn.Hardswish` | x * relu6(x+3)/6 | `inplace` |
| `nn.Hardsigmoid` | relu6(x+3)/6 | `inplace` |
| `nn.Hardtanh` | clamp(x, min_val, max_val) | `min_val=-1, max_val=1, inplace` |
| `nn.Sigmoid` | 1 / (1 + exp(-x)) | - |
| `nn.Tanh` | tanh(x) | - |
| `nn.Softmin` | softmax(-x) | `dim` |
| `nn.Softmax` | exp(x) / sum(exp(x)) | `dim` |
| `nn.Softmax2d` | Softmax over per-channel | - |
| `nn.LogSoftmax` | log(softmax(x)) | `dim` |
| `nn.Softplus` | 1/beta * log(1 + exp(beta*x)) | `beta=1, threshold=20` |
| `nn.Softsign` | x / (1 + \|x\|) | - |
| `nn.Tanhshrink` | x - tanh(x) | - |
| `nn.Softshrink` | soft threshold | `lambd=0.5` |
| `nn.Hardshrink` | hard threshold | `lambd=0.5` |
| `nn.Threshold` | y = x if x > threshold else value | `threshold, value, inplace` |

```python
# Common usage
relu = nn.ReLU()
gelu = nn.GELU()
silu = nn.SiLU()
softmax = nn.Softmax(dim=-1)
log_softmax = nn.LogSoftmax(dim=-1)
```

---

## 9.4 Weight and Spectral Normalization

```python
# Weight normalization: reparameterizes weight as g * (v / ||v||)
nn.utils.weight_norm(module, name='weight', dim=0)
nn.utils.remove_weight_norm(module, name='weight')

# Spectral normalization: constrains spectral norm of weight
nn.utils.spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=0)
nn.utils.remove_spectral_norm(module, name='weight')

# Usage
linear = nn.utils.spectral_norm(nn.Linear(10, 5))
conv = nn.utils.spectral_norm(nn.Conv2d(3, 64, 3))
```

---

## 9.5 Dropout Layers

```python
nn.Dropout(p=0.5, inplace=False)       # Standard dropout
nn.Dropout1d(p=0.5, inplace=False)     # Drop entire channels (1D)
nn.Dropout2d(p=0.5, inplace=False)     # Drop entire channels (2D)
nn.Dropout3d(p=0.5, inplace=False)     # Drop entire channels (3D)
nn.AlphaDropout(p=0.5, inplace=False)  # Maintains self-normalizing property (SELU)
```

During training: randomly zeroes elements with probability p. During eval: identity.
