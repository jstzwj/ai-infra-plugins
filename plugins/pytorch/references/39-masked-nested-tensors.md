# PyTorch Masked and Nested Tensors - Comprehensive Reference

This chapter covers masked tensors (`torch.masked`) and nested tensors (`torch.nested`), which handle variable-length sequences, padded data, and masked operations efficiently.

---

## 1. Masked Tensors

Masked tensors associate a mask with a tensor, marking certain elements as invalid or "masked out." Operations on masked tensors respect the mask, ignoring masked elements in reductions and computations.

### torch.masked.masked_tensor

Creates a masked tensor from a data tensor and a boolean mask.

```python
torch.masked.masked_tensor(
    data,       # (Tensor) the underlying data tensor
    mask,       # (Tensor) boolean mask (True = valid, False = masked)
    *, requires_grad=False,
)
```

```python
import torch

# Create a masked tensor
data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
mask = torch.tensor([True, True, False, True, False])

mt = torch.masked.masked_tensor(data, mask)
print(mt)
# masked_tensor([  1,   2, --,   4, --])

# Access underlying data and mask
print(mt.get_data())    # tensor([1., 2., 3., 4., 5.])
print(mt.get_mask())    # tensor([ True,  True, False,  True, False])

# Masked elements are ignored in operations
print(mt.sum())         # tensor(7.)  (1+2+4, ignoring 3 and 5)
print(mt.mean())        # tensor(2.3333)  ((1+2+4)/3)
print(mt.max())         # tensor(4.)
print(mt.min())         # tensor(1.)
```

### Creating Masked Tensors

```python
# From a tensor with NaN values
data = torch.tensor([1.0, float('nan'), 3.0, float('nan'), 5.0])
mask = ~torch.isnan(data)
mt = torch.masked.masked_tensor(data, mask)
print(mt)  # masked_tensor([  1, --,   3, --,   5])

# From a tensor with a specific fill value
data = torch.tensor([[1.0, 0.0, 3.0],
                      [0.0, 5.0, 0.0]])
mask = data != 0.0
mt = torch.masked.masked_tensor(data, mask)

# From a padded sequence
padded = torch.tensor([[1, 2, 3, 0, 0],
                        [4, 5, 0, 0, 0],
                        [6, 7, 8, 9, 0]])
lengths = torch.tensor([3, 2, 4])
mask = torch.arange(padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
mt = torch.masked.masked_tensor(padded.float(), mask)
```

---

## 2. Masked Operations

### Reduction Operations

```python
data = torch.tensor([[1.0, 2.0, 3.0, 0.0],
                      [5.0, 0.0, 7.0, 8.0]])
mask = torch.tensor([[True, True, True, False],
                      [True, False, True, True]])
mt = torch.masked.masked_tensor(data, mask)

# Sum (ignoring masked elements)
print(mt.sum())        # tensor(26.) (1+2+3+5+7+8)
print(mt.sum(dim=0))   # [6, 2, 10, 8] (column sums)
print(mt.sum(dim=1))   # [6, 20] (row sums)

# Mean
print(mt.mean())       # tensor(4.3333)
print(mt.mean(dim=0))  # column means (with valid count per column)
print(mt.mean(dim=1))  # row means

# Max and Min
print(mt.max())              # tensor(8.)
print(mt.max(dim=1))         # Row-wise max
print(mt.amin())             # tensor(1.)
print(mt.amax(dim=0))        # Column-wise max

# Prod
print(mt.prod())       # tensor(1680.) (1*2*3*5*7*8)

# Variance and Std
print(mt.var())        # variance over unmasked elements
print(mt.std())        # std over unmasked elements
```

### Element-wise Operations

```python
data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
mask = torch.tensor([True, True, False, True, False])
mt = torch.masked.masked_tensor(data, mask)

# Arithmetic operations
print(mt + 10)
# masked_tensor([ 11,  12, --,  14, --])

print(mt * 2)
# masked_tensor([  2,   4, --,   8, --])

print(mt ** 2)
# masked_tensor([  1,   4, --,  16, --])

# Unary operations
print(-mt)
print(mt.abs())
print(mt.sqrt())
print(mt.exp())
print(mt.log())
```

### Masked Tensor Comparison

```python
data1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
data2 = torch.tensor([1.0, 3.0, 2.0, 4.0])
mask = torch.tensor([True, True, False, True])

mt1 = torch.masked.masked_tensor(data1, mask)
mt2 = torch.masked.masked_tensor(data2, mask)

# Element-wise comparison (returns masked boolean tensor)
result = mt1 < mt2
print(result)  # masked_tensor([False,  True, --, False])

# Equality
eq = mt1 == mt2
print(eq)  # masked_tensor([ True, False, --,  True])
```

### Masked Softmax

```python
# Masked softmax: ignores masked elements and normalizes over valid ones
data = torch.tensor([[1.0, 2.0, 3.0, 0.0],
                      [4.0, 0.0, 5.0, 6.0]])
mask = torch.tensor([[True, True, True, False],
                      [True, False, True, True]])
mt = torch.masked.masked_tensor(data, mask)

# Softmax over dim=1 (each row normalizes independently)
softmax_result = mt.softmax(dim=1)
print(softmax_result)
# Row 0: [softmax(1), softmax(2), softmax(3), --]
# Row 1: [softmax(4), --, softmax(5), softmax(6)]

# Verify rows sum to 1 (over valid elements)
print(softmax_result.sum(dim=1))
```

### Masked Log-Sum-Exp

```python
# Numerically stable log-sum-exp over masked tensor
data = torch.tensor([[100.0, 101.0, 0.0],
                      [200.0, 0.0, 202.0]])
mask = torch.tensor([[True, True, False],
                      [True, False, True]])
mt = torch.masked.masked_tensor(data, mask)

result = mt.logsumexp(dim=1)
print(result)  # Stable computation ignoring masked values
```

### Where Operations

```python
# torch.masked.where
mask = torch.tensor([True, False, True, False])
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = torch.tensor([10.0, 20.0, 30.0, 40.0])

result = torch.masked.where(mask, x, y)
print(result)  # tensor([ 1., 20.,  3., 40.])
```

---

## 3. Masked Tensor with Autograd

```python
# Masked tensors support automatic differentiation
data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
mask = torch.tensor([True, True, False, True, True])
mt = torch.masked.masked_tensor(data, mask)

# Forward pass
result = (mt ** 2).sum()
print(result)  # 1 + 4 + 16 + 25 = 46 (3 is masked)

# Backward pass
result.backward()
print(data.grad)  # tensor([2., 4., 0., 8., 10.])
# Gradient is 0 for masked elements
```

---

## 4. Nested Tensors

Nested tensors represent a collection of tensors with different shapes but the same dtype. They are particularly useful for variable-length sequences (e.g., NLP, time series) where padding is wasteful.

### torch.nested.nested_tensor

```python
torch.nested.nested_tensor(
    tensor_list,        # (list of Tensors) tensors to nest
    *, dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False,
)
```

```python
import torch

# Create a nested tensor from a list of tensors with different sizes
tensors = [
    torch.randn(3, 5),   # Sequence of length 3
    torch.randn(5, 5),   # Sequence of length 5
    torch.randn(2, 5),   # Sequence of length 2
    torch.randn(7, 5),   # Sequence of length 7
]

nt = torch.nested.nested_tensor(tensors)
print(nt)
# NestedTensor([
#   tensor([[...], [...], [...]]),        # 3x5
#   tensor([[...], [...], [...], [...], [...]]),  # 5x5
#   tensor([[...], [...]]),               # 2x5
#   tensor([[...], [...], [...], [...], [...], [...], [...]]),  # 7x5
# ])

# Access individual tensors
print(nt[0].shape)   # torch.Size([3, 5])
print(nt[1].shape)   # torch.Size([5, 5])

# Get the size of each constituent tensor
print(nt.size())     # (3,5), (5,5), (2,5), (7,5)

# Check if a tensor is nested
print(nt.is_nested)  # True
```

### Nested Tensor from Padded Data

```python
# Create nested tensor from padded sequences
padded = torch.tensor([
    [1, 2, 3, 0, 0],   # Length 3
    [4, 5, 6, 7, 0],   # Length 4
    [8, 9, 0, 0, 0],   # Length 2
])
lengths = torch.tensor([3, 4, 2])

# Extract variable-length sequences
tensors = [padded[i, :lengths[i]] for i in range(len(lengths))]
nt = torch.nested.nested_tensor(tensors)

print(nt[0])  # tensor([1, 2, 3])
print(nt[1])  # tensor([4, 5, 6, 7])
print(nt[2])  # tensor([8, 9])
```

---

## 5. Nested Tensor Operations

### to_padded_tensor

Converts a nested tensor to a padded dense tensor.

```python
torch.nested.to_padded_tensor(
    nested_tensor,      # (NestedTensor) input nested tensor
    padding,            # (float) value to use for padding
    output_size=None,   # (tuple) desired output size
)
```

```python
# Create nested tensor
tensors = [
    torch.tensor([1.0, 2.0, 3.0]),
    torch.tensor([4.0, 5.0]),
    torch.tensor([6.0, 7.0, 8.0, 9.0]),
]
nt = torch.nested.nested_tensor(tensors)

# Convert to padded tensor
padded = torch.nested.to_padded_tensor(nt, padding=0.0)
print(padded)
# tensor([[1., 2., 3., 0.],
#         [4., 5., 0., 0.],
#         [6., 7., 8., 9.]])

# With explicit output size
padded = torch.nested.to_padded_tensor(nt, padding=0.0, output_size=[3, 6])
print(padded)
# tensor([[1., 2., 3., 0., 0., 0.],
#         [4., 5., 0., 0., 0., 0.],
#         [6., 7., 8., 9., 0., 0.]])
```

### Nested Tensor Concatenation

```python
# Concatenate nested tensors
nt1 = torch.nested.nested_tensor([
    torch.randn(3, 5),
    torch.randn(4, 5),
])
nt2 = torch.nested.nested_tensor([
    torch.randn(3, 5),
    torch.randn(4, 5),
])

# Concatenate along existing dimension
result = torch.cat([nt1, nt2], dim=0)
# Result has 4 constituent tensors
```

### Nested Tensor with Linear Layers

```python
# Nested tensors can be used with certain PyTorch modules
# This is particularly useful for Transformer models with variable-length sequences

nt = torch.nested.nested_tensor([
    torch.randn(3, 5),
    torch.randn(5, 5),
    torch.randn(2, 5),
])

# Apply a linear transformation
linear = torch.nn.Linear(5, 10)

# This works because the last dimension is consistent
# Each constituent tensor is transformed independently
result = linear(nt)
print(result[0].shape)  # torch.Size([3, 10])
print(result[1].shape)  # torch.Size([5, 10])
print(result[2].shape)  # torch.Size([2, 10])
```

### Nested Tensor with Attention

```python
# Efficient self-attention for variable-length sequences using nested tensors
# This avoids computing attention for padding tokens

def nested_self_attention(query, key, value):
    """
    Compute self-attention for nested tensors.
    query, key, value are NestedTensors with shapes (batch, seq_len, dim).
    """
    results = []
    for i in range(len(query)):
        q = query[i].unsqueeze(0)  # [1, seq_len, dim]
        k = key[i].unsqueeze(0)
        v = value[i].unsqueeze(0)

        # Scaled dot-product attention
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        results.append(attn.squeeze(0))

    return torch.nested.nested_tensor(results)
```

---

## 6. Use Cases

### Variable-Length Sequence Processing

```python
# Common use case: processing sentences of different lengths
sentences = [
    torch.randn(5, 768),    # 5 tokens, 768-dim embeddings
    torch.randn(12, 768),   # 12 tokens
    torch.randn(3, 768),    # 3 tokens
    torch.randn(8, 768),    # 8 tokens
]

# Without nested tensors: pad to max length (wastes compute)
max_len = max(s.size(0) for s in sentences)
padded = torch.zeros(len(sentences), max_len, 768)
for i, s in enumerate(sentences):
    padded[i, :s.size(0)] = s
# padded shape: [4, 12, 768] - lots of zeros

# With nested tensors: no padding needed
nt = torch.nested.nested_tensor(sentences)

# Apply transformer layer
transformer_layer = torch.nn.TransformerEncoderLayer(
    d_model=768, nhead=12, batch_first=True
)

# Process each sequence at its natural length
results = []
for i in range(len(nt)):
    out = transformer_layer(nt[i].unsqueeze(0))
    results.append(out.squeeze(0))

output = torch.nested.nested_tensor(results)
```

### Batched Variable-Length RNN

```python
class NestedRNN(torch.nn.Module):
    """RNN that handles variable-length sequences via nested tensors."""

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.rnn = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, nt):
        """Process a nested tensor of variable-length sequences."""
        outputs = []
        for i in range(len(nt)):
            seq = nt[i].unsqueeze(0)  # [1, seq_len, input_size]
            out, _ = self.rnn(seq)
            # Take last hidden state
            outputs.append(out.squeeze(0))

        return torch.nested.nested_tensor(outputs)

# Usage
sequences = [torch.randn(l, 64) for l in [5, 10, 3, 8, 12]]
nt = torch.nested.nested_tensor(sequences)

model = NestedRNN(input_size=64, hidden_size=128)
outputs = model(nt)
```

### Masked Loss for Variable-Length Targets

```python
def masked_sequence_loss(predictions, targets, lengths):
    """
    Compute loss for variable-length sequences using masking.

    Args:
        predictions: [batch, max_len, num_classes] logits
        targets: [batch, max_len] target indices
        lengths: [batch] actual sequence lengths
    """
    batch_size, max_len, num_classes = predictions.shape

    # Create mask from lengths
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.to(predictions.device)

    # Compute per-element loss
    log_probs = torch.special.log_softmax(predictions, dim=-1)
    per_element_loss = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # Apply mask
    masked_loss = per_element_loss * mask.float()

    # Average over valid elements
    total_loss = masked_loss.sum()
    num_valid = mask.sum()

    return total_loss / num_valid

# Usage
batch_size, max_len, num_classes = 4, 10, 100
predictions = torch.randn(batch_size, max_len, num_classes)
targets = torch.randint(0, num_classes, (batch_size, max_len))
lengths = torch.tensor([8, 5, 10, 3])  # Variable lengths

loss = masked_sequence_loss(predictions, targets, lengths)
print(loss)
```

---

## 7. Nested Tensor with Transformer

```python
class EfficientTransformer(torch.nn.Module):
    """Transformer that processes variable-length sequences efficiently."""

    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.pos_encoder = torch.nn.Embedding(1024, d_model)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, nt):
        """Process nested tensor input."""
        outputs = []
        for i in range(len(nt)):
            seq_len = nt[i].size(0)
            x = nt[i]  # [seq_len, d_model]

            # Add positional encoding
            positions = torch.arange(seq_len, device=x.device)
            x = x + self.pos_encoder(positions)

            # Apply transformer
            x = self.transformer(x.unsqueeze(0)).squeeze(0)
            outputs.append(x)

        return torch.nested.nested_tensor(outputs)
```

---

## 8. Comparison: Masked vs Nested vs Padded

| Aspect | Masked Tensors | Nested Tensors | Padded Tensors |
|--------|---------------|----------------|----------------|
| Memory | Same as padded + mask | Only valid elements | Wastes memory on padding |
| Compute | Wastes compute on padding | Only computes valid elements | Wastes compute on padding |
| API | Standard tensor ops with mask | Special nested API | Standard tensor ops |
| Flexibility | Arbitrary mask patterns | Must be list of tensors | Most flexible |
| Interop | Limited | Growing support | Full support |
| Gradient | Yes (zeros for masked) | Yes | Yes |
| Use case | Irregular valid regions | Variable-length sequences | Most common approach |

```python
# When to use which:
# - Padded tensors: default choice, most operations supported
# - Masked tensors: when you need to track valid/invalid elements
#   with arbitrary patterns (not just sequential padding)
# - Nested tensors: when you have variable-length sequences and
#   want to avoid padding waste entirely (e.g., in Transformers)
```

---

## 9. Advanced Patterns

### Masked Tensor for Sparse Labels

```python
def masked_label_loss(logits, labels, label_mask):
    """
    Compute loss only on specified labels.
    Useful for partial-label or multi-task learning.

    Args:
        logits: [batch, num_classes]
        labels: [batch, num_classes] target values
        label_mask: [batch, num_classes] which labels to include
    """
    mt_logits = torch.masked.masked_tensor(logits, label_mask.bool())
    mt_labels = torch.masked.masked_tensor(labels.float(), label_mask.bool())

    # Compute MSE only on masked elements
    diff = mt_logits - mt_labels
    loss = (diff ** 2).mean()
    return loss
```

### Nested Tensor Data Loading

```python
class VariableLengthDataset(torch.utils.data.Dataset):
    """Dataset that returns variable-length sequences."""

    def __init__(self, sequences, labels):
        self.sequences = sequences  # List of tensors with different lengths
        self.labels = labels

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

    def __len__(self):
        return len(self.sequences)

def nested_collate_fn(batch):
    """Collate function that creates nested tensors instead of padding."""
    sequences, labels = zip(*batch)
    nt = torch.nested.nested_tensor([s for s in sequences])
    labels = torch.tensor(labels)
    return nt, labels

# Usage
sequences = [torch.randn(l, 64) for l in [5, 10, 3, 8, 12, 6, 4, 7]]
labels = torch.randint(0, 10, (8,))

dataset = VariableLengthDataset(sequences, labels)
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    collate_fn=nested_collate_fn,
)

for nt_seq, batch_labels in loader:
    print(f"Nested tensor with {len(nt_seq)} sequences")
    for i in range(len(nt_seq)):
        print(f"  Sequence {i}: shape {nt_seq[i].shape}")
```

### Efficient Batch Processing with Nested Tensors

```python
def efficient_batch_forward(model, nested_input, chunk_size=4):
    """
    Process nested tensor in chunks for memory efficiency.
    Each chunk is converted to padded tensor for model forward pass,
    then results are collected back into nested tensor.
    """
    results = []
    for i in range(len(nested_input)):
        x = nested_input[i].unsqueeze(0)  # [1, seq_len, dim]
        out = model(x)
        results.append(out.squeeze(0))

    return torch.nested.nested_tensor(results)
```

### Nested Tensor to Padded with Mask

```python
def nested_to_padded_with_mask(nt, padding_value=0.0):
    """Convert nested tensor to padded tensor and return the mask."""
    padded = torch.nested.to_padded_tensor(nt, padding=padding_value)

    # Create mask from nested tensor sizes
    sizes = [nt[i].size(0) for i in range(len(nt))]
    max_len = padded.size(1)
    mask = torch.arange(max_len).unsqueeze(0) < torch.tensor(sizes).unsqueeze(1)

    return padded, mask

# Usage
nt = torch.nested.nested_tensor([
    torch.randn(3, 5),
    torch.randn(7, 5),
    torch.randn(2, 5),
])
padded, mask = nested_to_padded_with_mask(nt)
print(padded.shape)  # torch.Size([3, 7, 5])
print(mask)
# tensor([[True, True, True, False, False, False, False],
#         [True, True, True, True, True, True, True],
#         [True, True, False, False, False, False, False]])
```
