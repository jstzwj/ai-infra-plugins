# PyTorch Transforms - Comprehensive Reference

This chapter covers all transform utilities in `torchvision.transforms`, including image transforms, composition, functional API, custom transforms, and the v2 transform API.

---

## 1. Common Image Transforms

### Resize

Resize the input image to the given size.

```python
torchvision.transforms.Resize(
    size,                      # (int or sequence) desired output size
    interpolation=InterpolationMode.BILINEAR,  # interpolation method
    max_size=None,             # maximum allowed size for larger edge
    antialias=True,            # whether to use antialiasing
)
```

```python
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Resize to fixed size (smaller edge matched, larger edge scaled proportionally)
transform = transforms.Resize(256)

# Resize to specific height x width
transform = transforms.Resize((256, 256))

# Resize with different interpolation modes
transform = transforms.Resize(256, interpolation=InterpolationMode.BICUBIC)
transform = transforms.Resize(256, interpolation=InterpolationMode.NEAREST)
transform = transforms.Resize(256, interpolation=InterpolationMode.BILINEAR)
transform = transforms.Resize(256, interpolation=InterpolationMode.LANCZOS)

# With max_size constraint
transform = transforms.Resize(256, max_size=512)

# Apply to image
from PIL import Image
img = Image.open('photo.jpg')
resized = transform(img)
```

### CenterCrop

Crops the given image at the center to the given size.

```python
torchvision.transforms.CenterCrop(size)
```

```python
# Crop to 224x224 from center
transform = transforms.CenterCrop(224)

# Crop to specific height x width
transform = transforms.CenterCrop((256, 128))

# Crop to square from center
transform = transforms.CenterCrop(224)
img = Image.open('photo.jpg')
cropped = transform(img)  # 224x224 center crop
```

### RandomCrop

Crop the image at a random location.

```python
torchvision.transforms.RandomCrop(
    size,            # (int or sequence) desired output size
    padding=None,    # optional padding on each border
    pad_if_needed=False,  # pad if image is smaller than size
    fill=0,          # padding fill value
    padding_mode='constant',  # 'constant', 'edge', 'reflect', 'symmetric'
)
```

```python
# Random 224x224 crop
transform = transforms.RandomCrop(224)

# With padding
transform = transforms.RandomCrop(224, padding=4)

# With reflection padding
transform = transforms.RandomCrop(224, padding=4, padding_mode='reflect')

# Random crop with padding and fill color
transform = transforms.RandomCrop(224, padding=16, fill=128)
```

### RandomResizedCrop

Crop the image to a random size and aspect ratio, then resize to the given size.

```python
torchvision.transforms.RandomResizedCrop(
    size,                       # expected output size
    scale=(0.08, 1.0),         # range of fraction of area to crop
    ratio=(0.75, 1.3333),      # range of aspect ratio
    interpolation=InterpolationMode.BILINEAR,
    antialias=True,
)
```

```python
# Standard training augmentation (used in ImageNet training)
transform = transforms.RandomResizedCrop(224)

# More aggressive cropping
transform = transforms.RandomResizedCrop(
    224,
    scale=(0.5, 1.0),    # Crop at least 50% of area
    ratio=(0.75, 1.33),  # Near-square aspect ratios
)

# Less aggressive (useful for fine-grained tasks)
transform = transforms.RandomResizedCrop(
    224,
    scale=(0.8, 1.0),    # Crop at least 80% of area
)
```

### RandomHorizontalFlip

Horizontally flip the image randomly with a given probability.

```python
torchvision.transforms.RandomHorizontalFlip(p=0.5)
```

```python
# 50% chance of horizontal flip
transform = transforms.RandomHorizontalFlip(p=0.5)

# Always flip
transform = transforms.RandomHorizontalFlip(p=1.0)

# Never flip (identity transform)
transform = transforms.RandomHorizontalFlip(p=0.0)
```

### RandomVerticalFlip

Vertically flip the image randomly with a given probability.

```python
torchvision.transforms.RandomVerticalFlip(p=0.5)
```

```python
# For medical/satellite images where vertical flip makes sense
transform = transforms.RandomVerticalFlip(p=0.5)
```

### RandomRotation

Rotate the image by a random angle.

```python
torchvision.transforms.RandomRotation(
    degrees,           # range of rotation in degrees
    interpolation=InterpolationMode.NEAREST,
    expand=False,      # whether to expand the output
    center=None,       # center of rotation
    fill=0,            # fill value for new pixels
)
```

```python
# Rotate by up to 10 degrees in either direction
transform = transforms.RandomRotation(10)

# Rotate by 30 to 90 degrees
transform = transforms.RandomRotation((30, 90))

# Rotate with expansion to avoid cutting off content
transform = transforms.RandomRotation(15, expand=True)

# Rotate around a specific center
transform = transforms.RandomRotation(10, center=(100, 100))
```

### ColorJitter

Randomly change the brightness, contrast, saturation, and hue of an image.

```python
torchvision.transforms.ColorJitter(
    brightness=0,    # How much to jitter brightness
    contrast=0,      # How much to jitter contrast
    saturation=0,    # How much to jitter saturation
    hue=0,           # How much to jitter hue
)
```

```python
# Standard ImageNet augmentation
transform = transforms.ColorJitter(
    brightness=0.4,
    contrast=0.4,
    saturation=0.4,
    hue=0.1,
)

# Only brightness and contrast
transform = transforms.ColorJitter(brightness=0.2, contrast=0.2)

# Aggressive color augmentation
transform = transforms.ColorJitter(
    brightness=0.5,
    contrast=0.5,
    saturation=0.5,
    hue=0.2,
)
```

### Normalize

Normalize a tensor image with mean and standard deviation.

```python
torchvision.transforms.Normalize(
    mean,     # sequence of means for each channel
    std,      # sequence of standard deviations for each channel
    inplace=False,
)
```

```python
# ImageNet normalization (MUST apply after ToTensor)
transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

# For single-channel images (grayscale)
transform = transforms.Normalize(mean=[0.5], std=[0.5])

# Custom normalization (compute from your dataset)
# mean = dataset.mean(axis=(0, 2, 3))
# std = dataset.std(axis=(0, 2, 3))
transform = transforms.Normalize(mean=computed_mean, std=computed_std)
```

### ToTensor

Convert a PIL Image or numpy ndarray to tensor. Scales pixel values from [0, 255] to [0.0, 1.0].

```python
torchvision.transforms.ToTensor()
```

```python
from PIL import Image
import numpy as np

# Convert PIL Image to tensor
img = Image.open('photo.jpg')  # PIL Image HWC [0, 255]
transform = transforms.ToTensor()
tensor = transform(img)  # Tensor CHW [0.0, 1.0], float32

# Convert numpy array to tensor
arr = np.array(img)  # HWC uint8
tensor = transform(arr)  # CHW float32 [0.0, 1.0]
```

**Note:** `ToTensor` is deprecated in v2 transforms. Use `v2.ToImage()` followed by `v2.ToDtype(torch.float32, scale=True)` instead.

### ConvertImageDtype

Convert a tensor image to the given dtype, scaling values appropriately.

```python
torchvision.transforms.ConvertImageDtype(dtype)
```

```python
# Convert from uint8 [0, 255] to float32 [0.0, 1.0]
transform = transforms.ConvertImageDtype(torch.float32)

# Convert from float32 to uint8
transform = transforms.ConvertImageDtype(torch.uint8)
```

### Grayscale

Convert image to grayscale.

```python
torchvision.transforms.Grayscale(num_output_channels=1)
```

```python
# Single channel grayscale
transform = transforms.Grayscale(num_output_channels=1)

# Three channel grayscale (useful for pretrained models)
transform = transforms.Grayscale(num_output_channels=3)
```

### RandomGrayscale

Randomly convert image to grayscale.

```python
torchvision.transforms.RandomGrayscale(p=0.1)
```

```python
# 10% chance to convert to grayscale
transform = transforms.RandomGrayscale(p=0.1)
```

### GaussianBlur

Blur image with randomly chosen Gaussian blur.

```python
torchvision.transforms.GaussianBlur(
    kernel_size,          # Size of the Gaussian kernel
    sigma=(0.1, 2.0),    # Range for random sigma
)
```

```python
# Gaussian blur with 5x5 kernel
transform = transforms.GaussianBlur(kernel_size=5)

# With sigma range
transform = transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0))
```

### RandomAffine

Random affine transformation of the image.

```python
torchvision.transforms.RandomAffine(
    degrees,            # Range of rotation
    translate=None,     # Range of horizontal and vertical translations
    scale=None,         # Range of scaling factor
    shear=None,         # Range of shear
    interpolation=InterpolationMode.NEAREST,
    fill=0,
    center=None,
)
```

```python
# Random affine with rotation, translation, and scaling
transform = transforms.RandomAffine(
    degrees=10,
    translate=(0.1, 0.1),
    scale=(0.9, 1.1),
)

# With shear
transform = transforms.RandomAffine(
    degrees=15,
    shear=10,
)
```

### RandomPerspective

Perform a random perspective transformation.

```python
torchvision.transforms.RandomPerspective(
    distortion_scale=0.5,  # How much to distort
    p=0.5,                 # probability
    interpolation=InterpolationMode.BILINEAR,
    fill=0,
)
```

```python
# Random perspective distortion
transform = transforms.RandomPerspective(
    distortion_scale=0.5,
    p=0.5,
)
```

### RandomErasing

Randomly select a rectangle region in an image and erase its pixels.

```python
torchvision.transforms.RandomErasing(
    p=0.5,                  # probability of erasing
    scale=(0.02, 0.33),    # range of area ratio
    ratio=(0.3, 3.3),      # range of aspect ratio
    value=0,                # erasing value (int, str, or sequence)
    inplace=False,
)
```

```python
# Random erasing (applied to tensor, not PIL Image)
transform = transforms.RandomErasing(p=0.5, value='random')  # random noise

# Erase with specific value
transform = transforms.RandomErasing(p=0.5, value=0)  # black

# Erase with mean value
transform = transforms.RandomErasing(p=0.5, value=(0.485, 0.456, 0.406))
```

### Pad

Pad the image on all sides.

```python
torchvision.transforms.Pad(
    padding,                    # padding on each border
    fill=0,                     # fill value
    padding_mode='constant',    # 'constant', 'edge', 'reflect', 'symmetric'
)
```

```python
# Pad all sides by 4 pixels
transform = transforms.Pad(4)

# Pad each side differently: (left, top, right, bottom)
transform = transforms.Pad((4, 8, 4, 8))

# Reflective padding
transform = transforms.Pad(4, padding_mode='reflect')
```

### Lambda

Apply a user-defined lambda as a transform.

```python
torchvision.transforms.Lambda(lambd)
```

```python
# Apply custom function as transform
transform = transforms.Lambda(
    lambda x: x.rotate(90) if random.random() > 0.5 else x
)
```

---

## 2. Transform Composition

### Compose

Composes several transforms together.

```python
torchvision.transforms.Compose(transforms)
```

```python
# Standard ImageNet training transform pipeline
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

# Validation transform pipeline (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

# Apply
img = Image.open('photo.jpg')
augmented = train_transform(img)
```

### RandomApply

Apply a list of transformations randomly with a given probability.

```python
torchvision.transforms.RandomApply(
    transforms,     # list of transforms
    p=0.5,          # probability of applying
)
```

```python
# 50% chance of applying a sequence of color transforms
transform = transforms.RandomApply([
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.RandomGrayscale(p=0.2),
], p=0.5)

# Apply blur with 30% probability
transform = transforms.RandomApply([
    transforms.GaussianBlur(kernel_size=5),
], p=0.3)
```

### RandomChoice

Apply single transformation randomly picked from a list.

```python
torchvision.transforms.RandomChoice(transforms)
```

```python
# Apply exactly one of these transforms
transform = transforms.RandomChoice([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.RandomRotation(90),
])
```

### RandomOrder

Apply a list of transformations in a random order.

```python
torchvision.transforms.RandomOrder(transforms)
```

```python
# Apply all transforms but in random order
transform = transforms.RandomOrder([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2),
])
```

---

## 3. Functional Transforms

Functional transforms operate on tensors or PIL images directly without wrapping them in a class. They are the building blocks for class-based transforms.

```python
import torchvision.transforms.functional as F
```

### Core Functional Transforms

```python
from torchvision.transforms.functional import (
    resize, center_crop, crop, random_crop,
    hflip, vflip, rotate, affine, perspective,
    adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue, adjust_sharpness,
    normalize, to_tensor, to_pil_image,
    pad, gaussian_blur, invert, posterize, solarize, equalize,
    elastic_transform, five_crop, ten_crop,
)
```

### Example Usage

```python
import torchvision.transforms.functional as F
import torch

# Resize
img_tensor = F.resize(img_tensor, size=[224, 224])

# Center crop
img_tensor = F.center_crop(img_tensor, output_size=[224, 224])

# Horizontal flip
img_tensor = F.hflip(img_tensor)

# Normalize
img_tensor = F.normalize(
    img_tensor,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

# Adjust brightness (factor > 1 brightens, < 1 darkens)
img_tensor = F.adjust_brightness(img_tensor, brightness_factor=1.5)

# Adjust contrast
img_tensor = F.adjust_contrast(img_tensor, contrast_factor=1.2)

# Adjust saturation
img_tensor = F.adjust_saturation(img_tensor, saturation_factor=0.8)

# Adjust hue (factor in [-0.5, 0.5])
img_tensor = F.adjust_hue(img_tensor, hue_factor=0.1)

# Gaussian blur
img_tensor = F.gaussian_blur(img_tensor, kernel_size=[5, 5])

# Convert between formats
pil_img = F.to_pil_image(img_tensor)
tensor_img = F.to_tensor(pil_img)

# Rotation
img_tensor = F.rotate(img_tensor, angle=30.0)

# Affine transformation
img_tensor = F.affine(
    img_tensor,
    angle=0.0,
    translate=[10, 20],
    scale=1.0,
    shear=[0.0, 0.0],
)

# Perspective transform
img_tensor = F.perspective(img_tensor, startpoints, endpoints)

# Five crop (returns tuple of 5 crops: 4 corners + center)
crops = F.five_crop(img_tensor, size=[224, 224])

# Ten crop (5 crops + their horizontal flips)
crops = F.ten_crop(img_tensor, size=[224, 224])

# Pad
img_tensor = F.pad(img_tensor, padding=4, fill=0)

# Elastic transform
img_tensor = F.elastic_transform(
    img_tensor,
    displacement=displacement_map,
    interpolation=F.InterpolationMode.BILINEAR,
)
```

### Functional Transform with Bounding Boxes

```python
import torchvision.transforms.functional as F

def transform_with_bbox(image, bbox):
    """Apply transform to image and adjust bounding box accordingly."""
    # Horizontal flip both image and bbox
    if random.random() > 0.5:
        image = F.hflip(image)
        bbox[0], bbox[2] = image.width - bbox[2], image.width - bbox[0]

    # Resize both
    old_w, old_h = image.width, image.height
    image = F.resize(image, size=[224, 224])
    scale_x = 224.0 / old_w
    scale_y = 224.0 / old_h
    bbox[0] *= scale_x
    bbox[1] *= scale_y
    bbox[2] *= scale_x
    bbox[3] *= scale_y

    return image, bbox
```

---

## 4. Custom Transforms

### Basic Custom Transform

```python
class AddGaussianNoise:
    """Add Gaussian noise to a tensor."""

    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'

# Usage
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    AddGaussianNoise(mean=0.0, std=0.05),
])
```

### Custom Transform with Random State

```python
class RandomCutout:
    """Random cutout augmentation."""

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Image tensor of shape (C, H, W)
        Returns:
            Tensor: Image with cutout applied
        """
        h, w = img.size(1), img.size(2)
        mask = torch.ones((h, w), dtype=torch.float32)

        for _ in range(self.n_holes):
            y = torch.randint(0, h, (1,)).item()
            x = torch.randint(0, w, (1,)).item()

            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)

            mask[y1:y2, x1:x2] = 0.0

        mask = mask.expand_as(img)
        return img * mask
```

### Custom Transform with Parameters

```python
class MixUp:
    """Apply MixUp augmentation to a batch."""

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, batch_x, batch_y):
        """Apply MixUp to a batch."""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size)

        mixed_x = lam * batch_x + (1 - lam) * batch_x[index]
        y_a, y_b = batch_y, batch_y[index]

        return mixed_x, y_a, y_b, lam
```

### Custom Transform for Multi-Modal Data

```python
class MultiModalTransform:
    """Transform that operates on multiple modalities simultaneously."""

    def __init__(self, image_transform=None, text_transform=None):
        self.image_transform = image_transform
        self.text_transform = text_transform

    def __call__(self, sample):
        if self.image_transform and 'image' in sample:
            sample['image'] = self.image_transform(sample['image'])
        if self.text_transform and 'text' in sample:
            sample['text'] = self.text_transform(sample['text'])
        return sample
```

---

## 5. v2 Transforms

The v2 transforms API introduces a unified interface that works with images, bounding boxes, masks, and keypoints simultaneously.

### Importing v2

```python
from torchvision.transforms import v2

# Or import specific transforms
from torchvision.transforms.v2 import (
    Compose, Resize, CenterCrop, RandomCrop,
    RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip,
    RandomRotation, ColorJitter, Normalize, ToImage, ToDtype,
    RandomApply, RandomChoice, RandomOrder,
    GaussianBlur, RandomErasing, Pad,
    SanitizeBoundingBoxes,
)
```

### v2 Transform Pipeline

```python
from torchvision.transforms import v2

# v2 pipeline for classification
transforms_v2 = v2.Compose([
    v2.ToImage(),                              # Convert to tv_tensors.Image
    v2.Resize((256, 256)),                     # Resize
    v2.RandomCrop((224, 224)),                 # Random crop
    v2.RandomHorizontalFlip(),                 # Flip
    v2.ToDtype(torch.float32, scale=True),     # Convert to float, scale [0,255] -> [0,1]
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),   # Normalize
])
```

### v2 with Bounding Boxes and Masks

```python
from torchvision.transforms import v2
from torchvision import tv_tensors

# Define transforms that work on images, bounding boxes, and masks
transforms_v2 = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop((224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.SanitizeBoundingBoxes(),  # Remove degenerate boxes after transforms
])

# Apply to image + bounding boxes + masks
image = tv_tensors.Image(torch.randn(3, 256, 256))
bboxes = tv_tensors.BoundingBoxes(
    [[10, 20, 100, 200], [50, 60, 150, 250]],
    format=tv_tensors.BoundingBoxFormat.XYXY,
    canvas_size=(256, 256),
)
mask = tv_tensors.Mask(torch.randint(0, 2, (256, 256)))

# Transform all simultaneously
img_out, bbox_out, mask_out = transforms_v2(image, bboxes, mask)
```

### v2 Detection Pipeline

```python
from torchvision.transforms import v2
from torchvision import tv_tensors

# Detection-specific transform pipeline
train_transforms = v2.Compose([
    v2.ToImage(),
    v2.RandomPhotometricDistort(p=0.5),
    v2.RandomZoomOut(fill={tv_tensors.Image: (124, 116, 104), tv_tensors.Mask: 0}),
    v2.RandomIoUCrop(),
    v2.RandomHorizontalFlip(p=0.5),
    v2.SanitizeBoundingBoxes(min_size=1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Video transform pipeline (5D tensors)
video_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((256, 256)),
    v2.RandomCrop((224, 224)),
    v2.RandomHorizontalFlip(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### v2-Specific Transforms

```python
# RandomShortestSize (resize shorter edge to random value)
transform = v2.RandomShortestSize(
    min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
    max_size=1333,
)

# RandomIoUCrop (crop based on IoU with bounding boxes)
transform = v2.RandomIoUCrop()

# RandomPhotometricDistort (randomly change color properties)
transform = v2.RandomPhotometricDistort(
    brightness=(0.875, 1.125),
    contrast=(0.5, 1.5),
    saturation=(0.5, 1.5),
    hue=(-0.05, 0.05),
    p=0.5,
)

# ScaleJitter (randomly resize with jitter)
transform = v2.ScaleJitter(
    target_size=(512, 512),
    scale_range=(0.1, 2.0),
)

# RandomZoomOut (zoom out by placing image on larger canvas)
transform = v2.RandomZoomOut(
    fill={tv_tensors.Image: (124, 116, 104)},
    side_range=(1.0, 4.0),
    p=0.5,
)
```

---

## 6. Audio Transform Patterns

While `torchvision.transforms` focuses on images, similar patterns apply to audio using `torchaudio.transforms`.

```python
import torchaudio.transforms as T

# Spectrogram
spectrogram = T.Spectrogram(
    n_fft=1024,            # Size of FFT
    win_length=None,       # Window size (defaults to n_fft)
    hop_length=None,       # Hop length (defaults to win_length // 2)
    pad_mode='reflect',    # Padding mode
    power=2.0,             # Exponent for the magnitude (2.0 = power spectrogram)
)

# Mel Spectrogram
mel_spectrogram = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    win_length=1024,
    hop_length=512,
    f_min=0.0,
    f_max=None,
    n_mels=80,             # Number of mel filterbanks
    power=2.0,
)

# MFCC
mfcc = T.MFCC(
    sample_rate=16000,
    n_mfcc=40,             # Number of MFCC coefficients
    melkwargs={'n_fft': 1024, 'n_mels': 80, 'hop_length': 512},
)

# Time masking (SpecAugment)
time_masking = T.TimeMasking(time_mask_param=20)

# Frequency masking (SpecAugment)
freq_masking = T.FrequencyMasking(freq_mask_param=10)

# AmplitudeToDB (convert power/amplitude to decibels)
amp_to_db = T.AmplitudeToDB(stype='power', top_db=80.0)

# MuLaw encoding/decoding
mulaw_encode = T.MuLawEncoding(quantization_channels=256)
mulaw_decode = T.MuLawDecoding(quantization_channels=256)

# Resample
resampler = T.Resample(orig_freq=44100, new_freq=16000)

# GriffinLim (invert spectrogram to waveform)
griffin_lim = T.GriffinLim(
    n_fft=1024,
    n_iter=32,
    win_length=1024,
    hop_length=512,
)

# MelScale (convert frequency bins to mel scale)
mel_scale = T.MelScale(
    n_mels=80,
    sample_rate=16000,
    f_min=0.0,
    f_max=None,
)

# InverseMelScale
inv_mel_scale = T.InverseMelScale(
    n_mels=80,
    sample_rate=16000,
    f_min=0.0,
    f_max=None,
    n_stft=513,
)
```

### Audio Transform Pipeline

```python
# SpecAugment: combined masking pipeline
specaugment = torch.nn.Sequential(
    T.FrequencyMasking(freq_mask_param=15),
    T.TimeMasking(time_mask_param=35),
    T.TimeMasking(time_mask_param=35),  # Second time mask
)

# Full audio feature extraction pipeline
class AudioFeatureExtractor:
    def __init__(self, sample_rate=16000, n_mels=80):
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels,
        )
        self.amp_to_db = T.AmplitudeToDB()

    def __call__(self, waveform):
        mel = self.mel_transform(waveform)
        mel_db = self.amp_to_db(mel)
        return mel_db
```

---

## 7. Text Transform Patterns

While PyTorch does not have a dedicated text transform library, common patterns include:

```python
# Basic text preprocessing
import re

class TextPreprocessor:
    def __init__(self, lowercase=True, remove_punctuation=True):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation

    def __call__(self, text):
        if self.lowercase:
            text = text.lower()
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        return text

# Tokenization
class Tokenizer:
    def __init__(self, vocab, max_length=512):
        self.vocab = vocab
        self.max_length = max_length

    def __call__(self, text):
        tokens = text.split()[:self.max_length]
        ids = [self.vocab.get(t, self.vocab['<unk>']) for t in tokens]
        return torch.tensor(ids, dtype=torch.long)

# Text transform pipeline
class TextTransform:
    def __init__(self, vocab, max_length=512):
        self.preprocessor = TextPreprocessor()
        self.tokenizer = Tokenizer(vocab, max_length)

    def __call__(self, text):
        text = self.preprocessor(text)
        return self.tokenizer(text)
```

---

## 8. Common Transform Pipelines

### ImageNet Training Pipeline

```python
# Standard ImageNet training augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),
])

# Validation
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])
```

### SimCLR Augmentation

```python
class SimCLRTransform:
    """SimCLR self-supervised learning augmentations."""

    def __init__(self, size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5)
            ], p=0.5),
            transforms.RandomSolarize(threshold=128, p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        """Return two different augmented views of the same image."""
        return self.transform(x), self.transform(x)
```

### CutMix / MixUp Transform

```python
class CutMixCollator:
    """Collator that applies CutMix augmentation."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels)

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = images.size(0)
        index = torch.randperm(batch_size)

        # Generate random bounding box
        W, H = images.size(2), images.size(3)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(W, cx + cut_w // 2)
        y2 = min(H, cy + cut_h // 2)

        images[:, :, x1:x2, y1:y2] = images[index, :, x1:x2, y1:y2]
        lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)

        return images, labels, labels[index], lam
```

### Test-Time Augmentation (TTA)

```python
class TestTimeAugmentation:
    """Apply multiple augmentations at test time and average predictions."""

    def __init__(self, model, transform, num_augmentations=5):
        self.model = model
        self.transform = transform
        self.num_augmentations = num_augmentations

    def predict(self, image):
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for _ in range(self.num_augmentations):
                augmented = self.transform(image)
                augmented = augmented.unsqueeze(0).to(next(self.model.parameters()).device)
                pred = self.model(augmented)
                predictions.append(pred.softmax(dim=1))

        return torch.stack(predictions).mean(dim=0)
```

---

## 9. Auto-Augment and RandAugment

### AutoAugment

```python
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# ImageNet policy
transform = AutoAugment(policy=AutoAugmentPolicy.IMAGENET)

# CIFAR10 policy
transform = AutoAugment(policy=AutoAugmentPolicy.CIFAR10)

# SVHN policy
transform = AutoAugment(policy=AutoAugmentPolicy.SVHN)
```

### RandAugment

```python
from torchvision.transforms import RandAugment

# RandAugment with N=2 transforms, M=9 magnitude
transform = RandAugment(num_ops=2, magnitude=9)

# Usage in pipeline
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])
```

### TrivialAugmentWide

```python
from torchvision.transforms import TrivialAugmentWide

# Simplest augmentation strategy
transform = TrivialAugmentWide(num_magnitude_bins=31)
```

### AugMix

```python
from torchvision.transforms import AugMix

# AugMix augmentation
transform = AugMix(
    severity=3,
    mixture_width=3,
    chain_depth=-1,      # -1 = random depth
    alpha=1.0,
    all_ops=True,
)
```
