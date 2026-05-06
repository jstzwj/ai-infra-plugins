# 15 - jax.image

## Overview

`jax.image` provides image processing functions built on top of JAX, offering composable and differentiable image operations.

---

## 1. Resizing

### `jax.image.resize`

```python
jax.image.resize(image, shape, method, antialias=True, precision=None)
```

**Methods:**
- `"nearest"` — Nearest neighbor
- `"linear"` — Bilinear interpolation
- `"cubic"` — Bicubic interpolation
- `"lanczos3"` — Lanczos with a=3
- `"lanczos5"` — Lanczos with a=5

```python
import jax
import jax.numpy as jnp
from jax import image

# Resize a single image (H, W, C)
img = jnp.ones((256, 256, 3))
resized = image.resize(img, (128, 128, 3), method="bilinear")

# Resize a batch (N, H, W, C)
batch = jnp.ones((8, 256, 256, 3))
resized_batch = image.resize(batch, (8, 128, 128, 3), method="lanczos3")
```

### Resize with different methods comparison

```python
methods = ["nearest", "linear", "cubic", "lanczos3", "lanczos5"]
results = {m: image.resize(img, (64, 64, 3), method=m) for m in methods}
```

---

## 2. Scale and Translate

### `jax.image.scale_and_translate`

```python
jax.image.scale_and_translate(
    image,
    shape,
    spatial_dims,
    scale,
    translation,
    method,
    antialias=True,
    precision=None
)
```

```python
# Scale by 2x and shift by (10, 20) pixels
img = jnp.ones((100, 100, 3))
result = image.scale_and_translate(
    img,
    shape=(200, 200, 3),
    spatial_dims=(0, 1),
    scale=jnp.array([2.0, 2.0]),
    translation=jnp.array([10.0, 20.0]),
    method="linear"
)
```

---

## 3. Coordinate Utilities

### `jax.image.map_coordinates`

Map input array to new coordinates using interpolation.

```python
jax.image.map_coordinates(
    input,
    coordinates,
    order,
    mode='constant',
    cval=0.0
)
```

```python
# Rotate image using coordinate mapping
import math

img = jnp.arange(100).reshape(10, 10).astype(jnp.float32)

# Create rotated coordinate grid
h, w = img.shape
y, x = jnp.mgrid[0:h, 0:w]
theta = math.pi / 6  # 30 degrees

# Rotation matrix applied to coordinates
coords = jnp.stack([
    x * jnp.cos(theta) - y * jnp.sin(theta),
    x * jnp.sin(theta) + y * jnp.cos(theta)
])

rotated = image.map_coordinates(img, coords, order=1)
```

### Interpolation orders

| Order | Method |
|---|---|
| 0 | Nearest neighbor |
| 1 | Linear (bilinear for 2D) |
| 2 | Quadratic |
| 3 | Cubic (bicubic for 2D) |

### Boundary modes

| Mode | Behavior |
|---|---|
| `"constant"` | Fill with `cval` |
| `"nearest"` | Repeat edge values |
| `"wrap"` | Wrap around |
| `"reflect"` | Mirror at boundary |
| `"mirror"` | Mirror including boundary |

---

## 4. Affine Transforms

### Custom affine transform

```python
def affine_transform(img, matrix, output_shape):
    """Apply affine transform to image."""
    h, w = output_shape[:2]
    # Create coordinate grid for output
    out_coords = jnp.mgrid[0:h, 0:w].reshape(2, -1).astype(jnp.float32)

    # Add homogeneous coordinate
    out_coords_h = jnp.vstack([out_coords, jnp.ones((1, out_coords.shape[1]))])

    # Apply inverse transform to find source coordinates
    inv_matrix = jnp.linalg.inv(matrix)
    src_coords = inv_matrix @ out_coords_h

    # Map coordinates
    result = image.map_coordinates(
        img, src_coords[:2], order=1, mode='constant'
    )
    return result.reshape(output_shape)
```

---

## 5. Padding for Convolution

### `jax.image.pad_to_shape`

```python
# Pad image to target shape
img = jnp.ones((224, 224, 3))
padded = jnp.pad(img, [(0, 32), (0, 32), (0, 0)], mode='constant')
```

---

## 6. Data Augmentation with JIT

```python
@jax.jit
def augment_image(key, image):
    """Random augmentation pipeline."""
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Random horizontal flip
    flip = jax.random.bernoulli(k1)
    image = jnp.where(flip, jnp.flip(image, axis=1), image)

    # Random rotation via coordinate mapping
    angle = jax.random.uniform(k2, minval=-0.3, maxval=0.3)
    h, w = image.shape[:2]
    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
    y, x = jnp.mgrid[0:h, 0:w].astype(jnp.float32)
    cy, cx = h / 2.0, w / 2.0
    coords = jnp.stack([
        cos_a * (x - cx) + sin_a * (y - cy) + cx,
        -sin_a * (x - cx) + cos_a * (y - cy) + cy
    ])
    image = image.map_coordinates(image, coords, order=1, mode='nearest')

    # Random brightness
    brightness = jax.random.uniform(k3, minval=0.8, maxval=1.2)
    image = image * brightness

    # Random crop
    dy = jax.random.randint(k4, (), 0, 32)
    dx = jax.random.randint(k4, (), 0, 32)
    image = jax.lax.dynamic_slice(image, (dy, dx, 0), (h - 32, w - 32, image.shape[2]))

    return image
```

---

## 7. Differentiable Image Operations

Since all `jax.image` functions are JAX-native, they are automatically differentiable:

```python
def loss_fn(theta, img, target):
    """Differentiable rotation loss."""
    h, w = img.shape[:2]
    cos_t, sin_t = jnp.cos(theta), jnp.sin(theta)
    y, x = jnp.mgrid[0:h, 0:w].astype(jnp.float32)
    coords = jnp.stack([
        cos_t * x - sin_t * y,
        sin_t * x + cos_t * y
    ])
    rotated = image.map_coordinates(img, coords, order=1)
    return jnp.mean((rotated - target) ** 2)

# Gradient w.r.t. rotation angle
grad_rotation = jax.grad(loss_fn)(theta, img, target)
```

---

## 8. API Reference

```python
jax.image.resize(image, shape, method, *, antialias=True, precision=None)
jax.image.scale_and_translate(image, shape, spatial_dims, scale, translation, method, ...)
jax.image.map_coordinates(input, coordinates, order, mode='constant', cval=0.0)
```

### Resize methods

| Method | Quality | Speed | Use case |
|---|---|---|---|
| `"nearest"` | Low | Fastest | Segmentation masks |
| `"linear"` | Medium | Fast | General purpose |
| `"cubic"` | High | Medium | High-quality downsampling |
| `"lanczos3"` | Very high | Slow | Final output |
| `"lanczos5"` | Highest | Slowest | Maximum quality |
