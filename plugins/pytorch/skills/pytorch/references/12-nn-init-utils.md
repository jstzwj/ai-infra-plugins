# PyTorch - Chapter 12: nn.init and Utilities

This reference covers weight initialization, gradient clipping, RNN utilities, and other nn utility functions.

---

## 12.1 Weight Initialization

All initialization functions operate **in-place** (suffix `_`).

### Basic Initializers

```python
# Uniform distribution
torch.nn.init.uniform_(tensor, a=0.0, b=1.0)

# Normal distribution
torch.nn.init.normal_(tensor, mean=0.0, std=1.0)

# Constant value
torch.nn.init.constant_(tensor, val)

# All ones / zeros
torch.nn.init.ones_(tensor)
torch.nn.init.zeros_(tensor)

# Identity matrix
torch.nn.init.eye_(tensor)  # Only for 2D tensors

# Dirac delta (preserves activations in forward pass)
torch.nn.init.dirac_(tensor, groups=1)
```

### Xavier / Glorot Initialization

```python
# Uniform: U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out))
torch.nn.init.xavier_uniform_(tensor, gain=1.0)

# Normal: N(0, std) where std = gain * sqrt(2 / (fan_in + fan_out))
torch.nn.init.xavier_normal_(tensor, gain=1.0)
```

**Best for**: Sigmoid, Tanh activations.

### Kaiming / He Initialization

```python
# Uniform: U(-bound, bound) where bound = sqrt(6 / ((1 + a^2) * fan_in))
torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')

# Normal: N(0, std) where std = sqrt(2 / ((1 + a^2) * fan_in))
torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
```

- **a**: Negative slope for leaky_relu (default 0 for ReLU)
- **mode**: 'fan_in' (preserves variance in forward) or 'fan_out' (preserves variance in backward)
- **nonlinearity**: 'leaky_relu' or 'relu'

**Best for**: ReLU and variants (default initialization for nn.Linear and nn.Conv).

### Orthogonal Initialization

```python
torch.nn.init.orthogonal_(tensor, gain=1.0)
```

Fills with (semi-)orthogonal matrix. Good for RNNs.

### Sparse Initialization

```python
torch.nn.init.sparse_(tensor, sparsity, std=0.01)
```

Sets `sparsity` fraction of elements to zero, rest from N(0, std).

### Truncated Normal

```python
torch.nn.init.trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0)
```

Normal distribution truncated to [a, b]. Used in Vision Transformers.

### calculate_gain

```python
gain = torch.nn.init.calculate_gain(nonlinearity, param=None)
```

| nonlinearity | gain |
|---|---|
| linear / identity | 1 |
| conv{1,2,3}d | 1 |
| sigmoid | 1 |
| tanh | 5/3 ≈ 1.667 |
| relu | sqrt(2) ≈ 1.414 |
| leaky_relu | sqrt(2/(1+negative_slope^2)) |
| selu | 3/4 ≈ 0.75 |

---

## 12.2 Gradient Utilities

### clip_grad_norm_

```python
torch.nn.utils.clip_grad_norm_(
    parameters,           # Iterable of parameters or single parameter
    max_norm,             # Maximum norm value
    norm_type=2.0,        # p-norm type (float or 'inf')
    error_if_nonfinite=False,
)
# Returns: total_norm (float)
```

Clips gradient norm to max_norm. Most commonly used.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### clip_grad_value_

```python
torch.nn.utils.clip_grad_value_(parameters, clip_value, foreach=None)
```

Clips gradient values to [-clip_value, clip_value].

### clip_grad_norm_with_norm_ (newer)

```python
torch.nn.utils.clip_grad_norm_with_norm_(parameters, max_norm, total_norm, norm_type=2.0)
```

### parameters_to_vector / vector_to_parameters

```python
vec = torch.nn.utils.parameters_to_vector(parameters)  # Concat all params
torch.nn.utils.vector_to_parameters(vec, parameters)    # Split and assign
```

---

## 12.3 RNN Utilities

### pack_padded_sequence

```python
torch.nn.utils.rnn.pack_padded_sequence(
    input,                  # (T, B, *) or (B, T, *) if batch_first
    lengths,                # Sequence lengths
    batch_first=False,
    enforce_sorted=True,
)
```

### pad_packed_sequence

```python
torch.nn.utils.rnn.pad_packed_sequence(
    sequence,               # PackedSequence
    batch_first=False,
    padding_value=0.0,
    total_length=None,
)
```

### pad_sequence

```python
torch.nn.utils.rnn.pad_sequence(
    sequences,              # List of variable-length tensors
    batch_first=False,
    padding_value=0.0,
    padding_side='right',
)
```

### pack_sequence

```python
torch.nn.utils.rnn.pack_sequence(sequences, enforce_sorted=True)
```

---

## 12.4 Parametrize

```python
# Apply a reparameterization to a parameter
torch.nn.utils.parametrize.register_parametrization(
    module, tensor_name, parametrization
)

# Example: positive weight via exponential
class Positive(nn.Module):
    def forward(self, X):
        return torch.exp(X)

module = nn.Linear(10, 10)
parametrize.register_parametrization(module, "weight", Positive())

# Remove parametrization
torch.nn.utils.parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)

# Check if parametrized
torch.nn.utils.parametrize.is_parametrized(module, "weight")
```

---

## 12.5 skip_init

```python
# Create module without calling __init__ (for deferred initialization)
model = torch.nn.utils.skip_init(nn.Linear, 10, 5)
```

---

## 12.6 Weight/Spectral Norm (Functional)

```python
torch.nn.utils.weight_norm(module, name='weight', dim=0)
torch.nn.utils.remove_weight_norm(module, name='weight')
torch.nn.utils.spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=0)
torch.nn.utils.remove_spectral_norm(module, name='weight')
```

---

## 12.7 Memory Format Utilities

```python
# Convert to channels_last memory format
model = model.to(memory_format=torch.channels_last)
```
