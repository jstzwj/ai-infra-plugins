# PyTorch - Chapter 39: Masked and Nested Tensors

This reference covers masked tensors and nested (jagged) tensors.

---

## 39.1 Masked Tensors

```python
# Create masked tensor
data = torch.tensor([1.0, 2.0, 3.0, 4.0])
mask = torch.tensor([True, False, True, False])
mt = torch.masked.masked_tensor(data, mask)

# Operations respect the mask
mt.sum()          # Only sums masked=True elements: 1.0 + 3.0 = 4.0
mt.mean()         # Mean of masked elements: 2.0
```

---

## 39.2 Nested Tensors

```python
# Create from list of tensors with different sizes
tensors = [torch.randn(3, 10), torch.randn(5, 10), torch.randn(2, 10)]
nt = torch.nested.nested_tensor(tensors)

# Properties
nt.size(0)    # 3 (number of tensors)
nt.size(1)    # Variable (ragged dimension)

# Convert to padded
padded = nt.to_padded_tensor(padding=0.0, output_size=(5, 10))
# Shape: (3, 5, 10) - padded to max size

# Convert back
tensors_back = nt.to_tensor_list()
```

---

## 39.3 Use Cases

```python
# Attention with variable-length sequences
# No padding needed, more efficient
queries = torch.nested.nested_tensor([torch.randn(l, 64) for l in [10, 20, 15]])
keys = torch.nested.nested_tensor([torch.randn(l, 64) for l in [10, 20, 15]])
# Nested SDPA handles variable lengths efficiently
```
