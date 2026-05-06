# PyTorch - Chapter 11: nn.functional API

This reference covers the functional API (torch.nn.functional / F) for all nn operations without module wrappers.

```python
import torch.nn.functional as F
```

---

## 11.1 Convolution Functions

```python
F.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
F.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
F.conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)

F.conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1)
F.conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1)
F.conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1)
```

---

## 11.2 Pooling Functions

```python
F.avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
F.avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
F.avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)

F.max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)
F.max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)
F.max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

F.max_unpool1d(input, indices, kernel_size, stride=None, padding=0)
F.max_unpool2d(input, indices, kernel_size, stride=None, padding=0)
F.max_unpool3d(input, indices, kernel_size, stride=None, padding=0)

F.lp_pool1d(input, norm_type, kernel_size, stride=None, ceil_mode=False)
F.lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False)

F.adaptive_avg_pool1d(input, output_size)
F.adaptive_avg_pool2d(input, output_size)
F.adaptive_avg_pool3d(input, output_size)
F.adaptive_max_pool1d(input, output_size, return_indices=False)
F.adaptive_max_pool2d(input, output_size, return_indices=False)
F.adaptive_max_pool3d(input, output_size, return_indices=False)

F.fractional_max_pool2d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False)
F.fractional_max_pool3d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False)
```

---

## 11.3 Activation Functions

```python
F.relu(input, inplace=False)                # max(0, x)
F.relu6(input, inplace=False)               # min(max(0, x), 6)
F.leaky_relu(input, negative_slope=0.01, inplace=False)
F.prelu(input, weight)                       # Parametric ReLU
F.rrelu(input, lower=1./8, upper=1./3, training=False, inplace=False)
F.elu(input, alpha=1.0, inplace=False)
F.selu(input, inplace=False)                 # Scaled ELU
F.celu(input, alpha=1.0, inplace=False)
F.gelu(input, approximate='none')
F.silu(input, inplace=False)                 # x * sigmoid(x)
F.mish(input, inplace=False)                 # x * tanh(softplus(x))
F.hardswish(input, inplace=False)
F.hardsigmoid(input, inplace=False)
F.hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False)
F.sigmoid(input)                             # 1 / (1 + exp(-x))
F.hard_sigmoid(input)                        # Alias for hardsigmoid
F.tanh(input)
F.softmax(input, dim=None, _stacklevel=3, dtype=None)
F.log_softmax(input, dim=None, _stacklevel=3, dtype=None)
F.softmin(input, dim=None, _stacklevel=3, dtype=None)
F.softplus(input, beta=1, threshold=20)
F.softsign(input)
F.tanhshrink(input)
F.softshrink(input, lambd=0.5)
F.hardshrink(input, lambd=0.5)
F.threshold(input, threshold, value, inplace=False)
F.normalize(input, p=2.0, dim=1, eps=1e-12, out=None)
F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None)
```

---

## 11.4 Normalization Functions

```python
F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum=0.1, eps=1e-5)
F.layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5)
F.group_norm(input, num_groups, weight=None, bias=None, eps=1e-5)
F.instance_norm(input, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, momentum=0.1, eps=1e-5)
F.local_response_norm(input, size, alpha=1e-4, beta=0.75, k=1.0)
```

---

## 11.5 Dropout Functions

```python
F.dropout(input, p=0.5, training=True, inplace=False)
F.dropout1d(input, p=0.5, training=True, inplace=False)
F.dropout2d(input, p=0.5, training=True, inplace=False)
F.dropout3d(input, p=0.5, training=True, inplace=False)
F.alpha_dropout(input, p=0.5, training=True, inplace=False)
F.feature_alpha_dropout(input, p=0.5, training=True, inplace=False)
```

---

## 11.6 Linear Functions

```python
F.linear(input, weight, bias=None)
F.bilinear(input1, input2, weight, bias=None)
```

---

## 11.7 Sparse Functions

```python
F.embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0,
            scale_grad_by_freq=False, sparse=False)
F.embedding_bag(input, weight, offsets=None, max_norm=None, norm_type=2,
                scale_grad_by_freq=False, mode='mean', sparse=False,
                per_sample_weights=None, include_last_offset=False, padding_idx=None)
F.one_hot(input, num_classes=-1)
```

---

## 11.8 Loss Functions (Functional)

```python
F.cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100,
                reduce=None, reduction='mean', label_smoothing=0.0)
F.mse_loss(input, target, size_average=None, reduce=None, reduction='mean')
F.l1_loss(input, target, size_average=None, reduce=None, reduction='mean')
F.nll_loss(input, target, weight=None, size_average=None, ignore_index=-100,
           reduce=None, reduction='mean')
F.binary_cross_entropy(input, target, weight=None, size_average=None, reduce=None, reduction='mean')
F.binary_cross_entropy_with_logits(input, target, weight=None, size_average=None,
                                    reduce=None, reduction='mean', pos_weight=None)
F.kl_div(input, target, size_average=None, reduce=None, reduction='mean', log_target=False)
F.huber_loss(input, target, reduction='mean', delta=1.0)
F.smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='mean', beta=1.0)
F.cosine_embedding_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean')
F.margin_ranking_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean')
F.triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-6, swap=False,
                      size_average=None, reduce=None, reduction='mean')
F.triplet_margin_with_distance_loss(anchor, positive, negative, distance_function=None, margin=1.0, swap=False, reduction='mean')
F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False)
F.poisson_nll_loss(input, target, log_input=True, full=False, size_average=None, eps=1e-8, reduce=None, reduction='mean')
F.gaussian_nll_loss(input, variance, target, full=True, eps=1e-6, reduction='mean')
F.hinge_embedding_loss(input, target, margin=1.0, size_average=None, reduce=None, reduction='mean')
F.multi_margin_loss(input, target, p=1, margin=1.0, weight=None, size_average=None, reduce=None, reduction='mean')
F.multilabel_margin_loss(input, target, size_average=None, reduce=None, reduction='mean')
F.multilabel_soft_margin_loss(input, target, weight=None, size_average=None, reduce=None, reduction='mean')
F.soft_margin_loss(input, target, size_average=None, reduce=None, reduction='mean')
```

---

## 11.9 Vision Functions

```python
F.pad(input, pad, mode='constant', value=0)
# pad: (left, right, top, bottom) for 2D
# mode: 'constant', 'reflect', 'replicate', 'circular'

F.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None,
              recompute_scale_factor=None, antialias=False)

F.upsample(input, size=None, scale_factor=None, mode='nearest', align_corners=None)
F.upsample_nearest(input, size=None, scale_factor=None)
F.upsample_bilinear(input, size=None, scale_factor=None)

F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
F.affine_grid(theta, size, align_corners=True)

F.pixel_shuffle(input, upscale_factor)
F.pixel_unshuffle(input, downscale_factor)
```

---

## 11.10 Distance Functions

```python
F.pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False)
F.cosine_similarity(x1, x2, dim=1, eps=1e-8)
F.pdist(input, p=2.0)
```
