# PyTorch - Chapter 10: Loss Functions

This reference covers all loss functions in torch.nn with their formulas, parameters, input/output shapes, and examples.

---

## 10.1 Base Classes

```python
nn._Loss(size_average=None, reduce=None, reduction='mean')
```
- **reduction**: 'none' (no reduction), 'mean' (average), 'sum' (sum)

```python
nn._WeightedLoss(weight=None, size_average=None, reduce=None, reduction='mean')
```
- **weight**: Manual rescaling weight per class

---

## 10.2 Regression Losses

### nn.L1Loss (Mean Absolute Error)

```python
nn.L1Loss(reduction='mean')
```
Formula: `ℓ(x,y) = |x - y|`

### nn.MSELoss (Mean Squared Error)

```python
nn.MSELoss(reduction='mean')
```
Formula: `ℓ(x,y) = (x - y)²`

### nn.SmoothL1Loss (Huber Loss variant)

```python
nn.SmoothL1Loss(reduction='mean', beta=1.0)
```
Formula: `z = |x-y|; loss = 0.5*z²/beta if z < beta else z - 0.5*beta`

### nn.HuberLoss

```python
nn.HuberLoss(reduction='mean', delta=1.0)
```
Formula: `z = |x-y|; loss = 0.5*z² if z < delta else delta*(z-0.5*delta)`

---

## 10.3 Classification Losses

### nn.CrossEntropyLoss

```python
nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                    reduce=None, reduction='mean', label_smoothing=0.0)
```

Combines `LogSoftmax` + `NLLLoss`. Expects raw logits (NOT softmax outputs).

- **weight**: Per-class weight tensor
- **ignore_index**: Specifies target value to ignore
- **label_smoothing**: [0.0, 1.0] smoothing factor

**Input**: `(N, C)` or `(N, C, d1, d2, ...)` - raw logits
**Target**: `(N)` class indices or `(N, C)` probabilities

```python
loss = nn.CrossEntropyLoss()
logits = torch.randn(32, 10)    # batch=32, classes=10
target = torch.randint(0, 10, (32,))
loss(logits, target)
```

### nn.NLLLoss (Negative Log Likelihood)

```python
nn.NLLLoss(weight=None, size_average=None, ignore_index=-100,
           reduce=None, reduction='mean')
```

Expects log-probabilities (from LogSoftmax).

**Input**: `(N, C)` - log probabilities
**Target**: `(N)` - class indices

```python
m = nn.LogSoftmax(dim=1)
loss = nn.NLLLoss()
input = m(torch.randn(32, 10))
target = torch.randint(0, 10, (32,))
loss(input, target)
```

### nn.BCELoss (Binary Cross Entropy)

```python
nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
```

Expects probabilities (from Sigmoid).

**Input**: `(N, *)` - probabilities in [0, 1]
**Target**: `(N, *)` - same shape, values in [0, 1]

### nn.BCEWithLogitsLoss

```python
nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None,
                     reduction='mean', pos_weight=None)
```

Combines Sigmoid + BCELoss. Numerically more stable.

- **pos_weight**: Weight of positive examples (balances precision/recall)

```python
loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))
logits = torch.randn(32, 1)
target = torch.randint(0, 2, (32, 1)).float()
loss(logits, target)
```

### nn.KLDivLoss

```python
nn.KLDivLoss(size_average=None, reduce=None, reduction='mean', log_target=False)
```

KL divergence. Input should be log-probabilities.

### nn.NLLLoss2d (deprecated, use NLLLoss)

---

## 10.4 Ranking Losses

### nn.MarginRankingLoss

```python
nn.MarginRankingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
```

Given inputs x1, x2, label y ∈ {-1, 1}: `loss = max(0, -y*(x1-x2) + margin)`

### nn.TripletMarginLoss

```python
nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-6, swap=False,
                     size_average=None, reduce=None, reduction='mean')
```

`loss = max(d(anchor, positive) - d(anchor, negative) + margin, 0)`

```python
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
anchor = torch.randn(32, 128)
positive = torch.randn(32, 128)
negative = torch.randn(32, 128)
triplet_loss(anchor, positive, negative)
```

### nn.TripletMarginWithDistanceLoss

```python
nn.TripletMarginWithDistanceLoss(distance_function=None, margin=1.0,
                                 swap=False, reduction='mean')
```

### nn.CosineEmbeddingLoss

```python
nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
```

`loss = 1 - cos(x1, x2)` if y=1, else `max(0, cos(x1,x2) - margin)`

---

## 10.5 Other Losses

### nn.HingeEmbeddingLoss

```python
nn.HingeEmbeddingLoss(margin=1.0, size_average=None, reduce=None, reduction='mean')
```

### nn.MultiLabelMarginLoss (multi-class hinge)

```python
nn.MultiLabelMarginLoss(size_average=None, reduce=None, reduction='mean')
```

### nn.MultiLabelSoftMarginLoss

```python
nn.MultiLabelSoftMarginLoss(weight=None, size_average=None, reduce=None, reduction='mean')
```

### nn.MultiMarginLoss

```python
nn.MultiMarginLoss(p=1, margin=1.0, weight=None, size_average=None,
                   reduce=None, reduction='mean')
```

### nn.SoftMarginLoss

```python
nn.SoftMarginLoss(size_average=None, reduce=None, reduction='mean')
```

### nn.CTCLoss

```python
nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)
```

Connectionist Temporal Classification loss for sequence-to-sequence.

- **blank**: Blank label index
- **zero_infinity**: Zero out infinite loss and associated gradients

**Input**: `(T, N, C)` or `(T, C)` - log probabilities
**Target**: `(N, S)` or `(sum(target_lengths))` - concatenated targets

### nn.PoissonNLLLoss

```python
nn.PoissonNLLLoss(log_input=True, full=False, size_average=None, eps=1e-8,
                  reduce=None, reduction='mean')
```

### nn.GaussianNLLLoss

```python
nn.GaussianNLLLoss(full=True, eps=1e-6, reduction='mean')
```

Expects input (mean) and variance. Target is observation.
