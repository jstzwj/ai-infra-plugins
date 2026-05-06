# PyTorch Transforms - Comprehensive Reference

This chapter covers data transforms for PyTorch, including torchvision transforms (v1 and v2 APIs), functional transforms, custom transforms, and integration patterns with data loading pipelines.

---

## 1. torchvision.transforms Overview

The `torchvision.transforms` module provides common image transformations for data augmentation and preprocessing. These are used with `torch.utils.data.DataLoader` to transform data on-the-fly during training.

### Basic Pipeline

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

---

## 2. Image Transforms (v1 API)

### Geometric Transforms

#### Resize(size, interpolation, max_size, antialias)

Resizes the image to the given size.

```python
transforms.Resize(256)                                    # Shortest side to 256
transforms.Resize((256, 256))                             # Exact height x width
transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC)
transforms.Resize(256, antialias=True)                    # Recommended for downscaling
```

**Parameters:**
- `size` (int or sequence): Desired output size. If int, smaller edge matched; if sequence (h, w), exact size.
- `interpolation` (InterpolationMode): `NEAREST`, `BILINEAR`, `BICUBIC` (default).
- `max_size` (int): Maximum size of the longer edge.
- `antialias` (bool): Whether to apply antialiasing (default True).

#### CenterCrop(size)

Crops the center of the image.

```python
transforms.CenterCrop(224)           # 224x224 center crop
transforms.CenterCrop((300, 400))    # 300x400 center crop
```

#### RandomCrop(size, padding, pad_if_needed, fill, padding_mode)

Crops the image at a random location.

```python
transforms.RandomCrop(224)
transforms.RandomCrop(224, padding=4)                # Pad 4 pixels then crop
transforms.RandomCrop(224, padding=4, padding_mode='reflect')
transforms.RandomCrop(224, pad_if_needed=True)
```

**Parameters:**
- `size` (int or sequence): Desired output size.
- `padding` (int or sequence): Optional padding on each border.
- `pad_if_needed` (bool): Pad if image is smaller than desired size.
- `fill` (int or tuple): Pixel fill value for padding (default 0).
- `padding_mode` (str): `constant`, `edge`, `reflect`, `symmetric`.

#### RandomResizedCrop(size, scale, ratio, interpolation, antialias)

Crops a random portion of the image and resizes it.

```python
transforms.RandomResizedCrop(224)
transforms.RandomResizedCrop(
    224,
    scale=(0.08, 1.0),      # Random area of the crop
    ratio=(3.0/4, 4.0/3),   # Random aspect ratio
    interpolation=transforms.InterpolationMode.BICUBIC,
)
```

#### RandomHorizontalFlip(p=0.5)

Horizontally flips the image randomly with a given probability.

```python
transforms.RandomHorizontalFlip(p=0.5)
```

#### RandomVerticalFlip(p=0.5)

Vertically flips the image randomly.

```python
transforms.RandomVerticalFlip(p=0.5)
```

#### RandomAffine(degrees, translate, scale, shear, interpolation, fill)

Random affine transformation of the image.

```python
transforms.RandomAffine(degrees=30)                              # Rotation only
transforms.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1))
transforms.RandomAffine(degrees=0, scale=(0.8, 1.2))
transforms.RandomAffine(degrees=0, shear=(-10, 10))
transforms.RandomAffine(degrees=30, fill=128)                   # Fill color
```

#### RandomRotation(degrees, interpolation, expand, center, fill)

Rotates the image by a random angle.

```python
transforms.RandomRotation(30)                    # -30 to +30 degrees
transforms.RandomRotation((-45, 45))             # Custom range
transforms.RandomRotation(30, expand=True)       # Expand to fit rotated image
transforms.RandomRotation(30, center=(112, 112)) # Custom center
```

#### RandomPerspective(distortion_scale, p, interpolation, fill)

Performs a random perspective transformation.

```python
transforms.RandomPerspective(distortion_scale=0.5, p=0.5)
```

#### RandomErasing(p, scale, ratio, value, inplace)

Randomly selects a rectangle region and erases its pixels.

```python
transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
transforms.RandomErasing(p=0.5, value='random')  # Random noise
```

#### ElasticTransform(alpha, sigma, interpolation, fill)

Applies elastic transformation (useful for medical imaging).

```python
transforms.ElasticTransform(alpha=50.0, sigma=5.0)
```

### Color Transforms

#### ColorJitter(brightness, contrast, saturation, hue)

Randomly changes brightness, contrast, saturation, and hue.

```python
transforms.ColorJitter(brightness=0.2)
transforms.ColorJitter(contrast=0.2)
transforms.ColorJitter(saturation=0.2)
transforms.ColorJitter(hue=0.1)
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
```

#### GaussianBlur(kernel_size, sigma)

Adds Gaussian blur with random kernel size and sigma.

```python
transforms.GaussianBlur(kernel_size=5)
transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0))
```

#### RandomGrayscale(p, num_output_channels)

Converts image to grayscale with probability p.

```python
transforms.RandomGrayscale(p=0.1)
transforms.RandomGrayscale(p=0.1, num_output_channels=3)  # Keep 3 channels
```

#### RandomInvert(p)

Inverts the colors of the image.

```python
transforms.RandomInvert(p=0.5)
```

#### RandomPosterize(bits, p)

Posterizes the image by reducing bit depth.

```python
transforms.RandomPosterize(bits=4, p=0.5)
```

#### RandomSolarize(threshold, p)

Solarizes the image by inverting pixels above a threshold.

```python
transforms.RandomSolarize(threshold=128, p=0.5)
```

#### RandomAdjustSharpness(sharpness_factor, p)

Adjusts the sharpness of the image.

```python
transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5)
```

#### RandomAutoContrast(p)

Applies auto-contrast.

```python
transforms.RandomAutoContrast(p=0.5)
```

#### RandomEqualize(p)

Equalizes the image histogram.

```python
transforms.RandomEqualize(p=0.5)
```

### Conversion Transforms

#### ToTensor

Converts a PIL Image or NumPy ndarray to a tensor. Scales pixel values from [0, 255] to [0.0, 1.0].

```python
transforms.ToTensor()
```

**Note:** In v2, `ToTensor` is deprecated. Use `v2.ToImage()` followed by `v2.ToDtype()`.

#### PILToTensor

Converts a PIL Image to a tensor without scaling.

```python
transforms.PILToTensor()
```

#### ConvertImageDtype(dtype)

Converts tensor image to the given dtype and scales values appropriately.

```python
transforms.ConvertImageDtype(torch.float32)
```

### Normalization

#### Normalize(mean, std, inplace)

Normalizes a tensor image with mean and standard deviation.

```python
# ImageNet normalization
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

# Per-channel normalization
transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
)
```

**Formula:** `output = (input - mean) / std`

### Composition Transforms

#### Compose(transforms)

Composes several transforms together.

```python
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

#### RandomApply(transforms, p)

Applies a list of transforms randomly with a given probability.

```python
transforms.RandomApply([
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
], p=0.5)
```

#### RandomChoice(transforms)

Applies a single transform randomly picked from a list.

```python
transforms.RandomChoice([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])
```

#### RandomOrder(transforms)

Applies transforms in a random order.

```python
transforms.RandomOrder([
    transforms.RandomRotation(30),
    transforms.ColorJitter(0.2),
    transforms.GaussianBlur(5),
])
```

### Linear Transformation

#### LinearTransformation(transformation_matrix, mean_vector)

Applies a linear transformation to the image (e.g., ZCA whitening).

```python
# ZCA whitening
transform = transforms.LinearTransformation(
    transformation_matrix=zca_matrix,  # Computed from training data
    mean_vector=mean_vector,
)
```

---

## 3. Functional Transforms (torchvision.transforms.functional)

Functional transforms operate on tensors or PIL images directly without creating transform objects. They are useful for custom augmentation logic.

```python
import torchvision.transforms.functional as F
```

### Image Operations

```python
# Resize
resized = F.resize(img, size=[256], interpolation=F.InterpolationMode.BICUBIC)

# Center crop
cropped = F.center_crop(img, output_size=[224])

# Random crop (requires explicit parameters)
cropped, top, left = F.crop(img, top=10, left=10, height=224, width=224)

# Horizontal/Vertical flip
flipped = F.hflip(img)
flipped = F.vflip(img)

# Rotate
rotated = F.rotate(img, angle=45)

# Affine
affined = F.affine(img, angle=0, translate=[10, 10], scale=1.0, shear=[0])

# Perspective
perspectived = F.perspective(img, startpoints, endpoints)

# Normalize
normalized = F.normalize(tensor, mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])

# Adjust brightness/contrast/saturation/hue
bright = F.adjust_brightness(img, brightness_factor=1.5)
contrast = F.adjust_contrast(img, contrast_factor=2.0)
saturated = F.adjust_saturation(img, saturation_factor=0.5)
hued = F.adjust_hue(img, hue_factor=0.1)

# Gaussian blur
blurred = F.gaussian_blur(img, kernel_size=[5, 5])

# Convert
tensor = F.to_tensor(img)
pil = F.to_pil_image(tensor)

# Pad
padded = F.pad(img, padding=4, fill=0, padding_mode='constant')

# Crop
cropped = F.crop(img, top=0, left=0, height=224, width=224)

# Erase
erased = F.erase(img, i=10, j=10, h=50, w=50, v=0)
```

### Functional with Bounding Boxes and Masks

```python
# Bounding box operations (v2)
from torchvision.transforms import v2 as T_v2

# These also transform bounding boxes and masks:
cropped = T_v2.RandomCrop(224)
resized = T_v2.Resize(256)
flipped = T_v2.RandomHorizontalFlip()
```

---

## 4. v2 Transforms API

The v2 API (introduced in torchvision 0.15) provides several improvements:

1. **Support for videos, bounding boxes, masks, and key points** in addition to images.
2. **Better performance** through native tensor operations.
3. **Simplified pipeline** with fewer transforms needed.

### v2 Transform Classes

```python
from torchvision.transforms import v2

transform = v2.Compose([
    v2.ToImage(),                           # Convert to tv_tensors.Image
    v2.RandomResizedCrop(224, antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomAutoContrast(p=0.2),
    v2.ToDtype(torch.float32, scale=True),  # Scale [0,255] -> [0,1]
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])
```

### v2 with Bounding Boxes and Masks

```python
from torchvision.transforms import v2
from torchvision import tv_tensors

# v2 transforms handle bounding boxes and masks correctly
transform = v2.Compose([
    v2.RandomResizedCrop(224, antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])

# Apply to image + bboxes + mask
image = tv_tensors.Image(torch.rand(3, 300, 400))
bboxes = tv_tensors.BoundingBoxes(
    [[10, 20, 100, 200], [50, 60, 150, 250]],
    format=tv_tensors.BoundingBoxFormat.XYXY,
    canvas_size=(300, 400),
)
mask = tv_tensors.Mask(torch.randint(0, 2, (300, 400)))

# All are transformed consistently
image_t, bboxes_t, mask_t = transform(image, bboxes, mask)
```

### v2 CutMix and MixUp

```python
from torchvision.transforms import v2

cutmix = v2.CutMix(num_classes=10, alpha=1.0)
mixup = v2.MixUp(num_classes=10, alpha=0.2)

# Apply in DataLoader collate_fn
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    images, labels = cutmix(images, labels)
    return images, labels
```

### v2 AutoAugment Policies

```python
from torchvision.transforms import v2

# AutoAugment with ImageNet policy
transform = v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET)

# RandAugment
transform = v2.RandAugment(num_ops=2, magnitude=9)

# TrivialAugmentWide
transform = v2.TrivialAugmentWide()

# AugMix
transform = v2.AugMix(severity=3, mixture_width=3, chain_depth=-1)
```

---

## 5. Text Transforms

PyTorch does not include built-in text transforms in torchvision, but common patterns are used with the torchtext library.

### Tokenization and Vocabulary

```python
# Using torchtext (common PyTorch ecosystem pattern)
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(
    [tokenizer(text) for text in corpus],
    specials=['<unk>', '<pad>', '<bos>', '<eos>'],
)
vocab.set_default_index(vocab['<unk>'])

# Custom text transform
class TextTransform:
    def __init__(self, vocab, tokenizer, max_len=256):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, text):
        tokens = self.tokenizer(text)[:self.max_len]
        indices = [self.vocab['<bos>']] + \
                  [self.vocab[t] for t in tokens] + \
                  [self.vocab['<eos>']]
        return torch.tensor(indices, dtype=torch.long)
```

### Custom Text Pipeline

```python
class TextPipeline:
    def __init__(self, vocab, tokenizer, max_length=128):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, text):
        tokens = self.tokenizer(text)
        # Truncate
        tokens = tokens[:self.max_length - 2]
        # Add special tokens
        token_ids = (
            [self.vocab['<bos>']]
            + [self.vocab.get(t, self.vocab['<unk>']) for t in tokens]
            + [self.vocab['<eos>']]
        )
        # Pad
        padding_length = self.max_length - len(token_ids)
        token_ids = token_ids + [self.vocab['<pad>']] * padding_length
        # Attention mask
        mask = [1] * (len(tokens) + 2) + [0] * padding_length
        return {
            'input_ids': torch.tensor(token_ids),
            'attention_mask': torch.tensor(mask),
        }
```

---

## 6. Audio Transforms

Audio transforms are available through torchaudio, the PyTorch audio library.

### Spectrogram Transforms

```python
import torchaudio.transforms as T

# Spectrogram
spec_transform = T.Spectrogram(
    n_fft=1024,
    win_length=1024,
    hop_length=512,
    power=2.0,
)

# Mel Spectrogram
mel_transform = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    win_length=1024,
    hop_length=512,
    n_mels=80,
    f_min=0.0,
    f_max=8000.0,
)

# MFCC
mfcc_transform = T.MFCC(
    sample_rate=16000,
    n_mfcc=40,
    melkwargs={
        'n_fft': 1024,
        'n_mels': 80,
        'hop_length': 512,
    },
)

# Amplitude to DB
amp_to_db = T.AmplitudeToDB(stype='power', top_db=80)

# Mu-law encoding/decoding
mu_law_encode = T.MuLawEncoding(quantization_channels=256)
mu_law_decode = T.MuLawDecoding(quantization_channels=256)
```

### Audio Augmentation

```python
import torchaudio.transforms as T

# Frequency Masking (SpecAugment)
freq_mask = T.FrequencyMasking(freq_mask_param=30)

# Time Masking (SpecAugment)
time_mask = T.TimeMasking(time_mask_param=100)

# Time Stretch
time_stretch = T.TimeStretch(
    hop_length=512,
    n_freq=513,
    fixed_rate=1.2,
)

# Pitch Shift
pitch_shift = T.PitchShift(
    sample_rate=16000,
    n_steps=4,
)

# Resampling
resample = T.Resample(
    orig_freq=44100,
    new_freq=16000,
    resampling_method='sinc_interpolation',
)

# Vad (Voice Activity Detection)
vad = T.Vad(sample_rate=16000)
```

### Audio Transform Pipeline

```python
class AudioTransformPipeline:
    def __init__(self, sample_rate=16000, n_mels=80):
        self.spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels,
        )
        self.amp_to_db = T.AmplitudeToDB()
        self.freq_mask = T.FrequencyMasking(freq_mask_param=30)
        self.time_mask = T.TimeMasking(time_mask_param=100)

    def __call__(self, waveform, augment=False):
        spec = self.amp_to_db(self.spectrogram(waveform))
        if augment:
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)
        return spec
```

---

## 7. Custom Transforms

### Basic Custom Transform

```python
class AddGaussianNoise:
    """Add Gaussian noise to a tensor."""

    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'
```

### Custom Transform with Random State

```python
class RandomGaussianBlur:
    """Gaussian blur with configurable probability."""

    def __init__(self, kernel_size=5, sigma=(0.1, 2.0), p=0.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            sigma = torch.uniform(self.sigma[0], self.sigma[1]).item()
            return F.gaussian_blur(img, [self.kernel_size, self.kernel_size], [sigma])
        return img
```

### Custom Transform with Parameters

```python
class Cutout:
    """Random cutout augmentation."""

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Image tensor of shape (C, H, W).
        Returns:
            Tensor: Image with cutout applied.
        """
        h, w = img.shape[1], img.shape[2]
        mask = torch.ones((h, w), dtype=torch.float32)

        for _ in range(self.n_holes):
            y = torch.randint(0, h, (1,)).item()
            x = torch.randint(0, w, (1,)).item()
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            mask[y1:y2, x1:x2] = 0

        mask = mask.expand_as(img)
        return img * mask
```

### Lambda Transform

```python
# Inline transform using Lambda
transform = transforms.Lambda(lambda x: x * 2.0 + 0.5)

# With a function
def normalize_range(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

transform = transforms.Lambda(normalize_range)
```

### Stateful Transform

```python
class RunningNormalize:
    """Normalize using running statistics (no pre-computed mean/std)."""

    def __init__(self, num_features, momentum=0.1):
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.num_batches_tracked = 0

    def __call__(self, tensor):
        if self.training:
            mean = tensor.mean(dim=0)
            var = tensor.var(dim=0, unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                                self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                               self.momentum * var
            self.num_batches_tracked += 1
        return (tensor - self.running_mean) / torch.sqrt(self.running_var + 1e-5)
```

---

## 8. Transform Pipelines and Augmentation Strategies

### Standard Training Pipeline

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                           saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

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
    """SimCLR data augmentation for contrastive learning."""

    def __init__(self, size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(5)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        """Return two augmented views of the same image."""
        return self.transform(x), self.transform(x)
```

### BYOL / MoCo Augmentation

```python
class BYOLTransform:
    """BYOL augmentation: strong + weak views."""

    def __init__(self, size=224):
        self.weak = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self.strong = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(5)], p=0.5),
            transforms.RandomSolarize(128, p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        return self.strong(x), self.weak(x)
```

### FixRes Augmentation

```python
# Different train and test resolutions
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize(384),                    # Higher resolution at test
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

---

## 9. DataLoader with Transforms

### Standard Pattern

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(
    'data/train',
    transform=train_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
```

### GPU-Based Transforms with Kornia

```python
import kornia as K

class GPUTransform(nn.Module):
    """Apply augmentations on GPU for better performance."""

    def __init__(self):
        super().__init__()
        self.augmentation = K.AugmentationSequential(
            K.augmentation.RandomResizedCrop((224, 224)),
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.5),
            data_keys=["input"],
        )

    def forward(self, x):
        return self.augmentation(x)

# CPU: minimal transform (just ToTensor and Normalize)
cpu_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# GPU augmentation
gpu_augment = GPUTransform().to('cuda')

for images, labels in train_loader:
    images = images.to('cuda')
    images = gpu_augment(images)  # Augment on GPU
    output = model(images)
```

---

## 10. Albumentations Integration

Albumentations is a fast image augmentation library that integrates well with PyTorch.

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define augmentation pipeline
transform = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.08, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                       rotate_limit=15, p=0.5),
    A.ColorJitter(brightness=0.4, contrast=0.4,
                  saturation=0.4, hue=0.1, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# With bounding boxes
transform_with_bboxes = A.Compose([
    A.RandomResizedCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# Dataset integration
class AlbumentationsDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.image_dir, self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image
```

---

## 11. Transform Summary Table

### v1 Transforms

| Transform | Input | Output | Category |
|-----------|-------|--------|----------|
| `Resize` | PIL/Tensor | PIL/Tensor | Geometric |
| `CenterCrop` | PIL/Tensor | PIL/Tensor | Geometric |
| `RandomCrop` | PIL/Tensor | PIL/Tensor | Geometric |
| `RandomResizedCrop` | PIL/Tensor | PIL/Tensor | Geometric |
| `RandomHorizontalFlip` | PIL/Tensor | PIL/Tensor | Geometric |
| `RandomVerticalFlip` | PIL/Tensor | PIL/Tensor | Geometric |
| `RandomAffine` | PIL/Tensor | PIL/Tensor | Geometric |
| `RandomRotation` | PIL/Tensor | PIL/Tensor | Geometric |
| `RandomPerspective` | PIL/Tensor | PIL/Tensor | Geometric |
| `RandomErasing` | Tensor | Tensor | Geometric |
| `ColorJitter` | PIL/Tensor | PIL/Tensor | Color |
| `GaussianBlur` | PIL/Tensor | PIL/Tensor | Color |
| `RandomGrayscale` | PIL/Tensor | PIL/Tensor | Color |
| `Normalize` | Tensor | Tensor | Conversion |
| `ToTensor` | PIL/ndarray | Tensor | Conversion |
| `PILToTensor` | PIL | Tensor | Conversion |
| `Compose` | Any | Any | Composition |
| `RandomApply` | Any | Any | Composition |
| `RandomChoice` | Any | Any | Composition |
| `RandomOrder` | Any | Any | Composition |
| `LinearTransformation` | Tensor | Tensor | Mathematical |

### v2 Key Differences

| Feature | v1 | v2 |
|---------|----|----|
| Bounding box support | No | Yes |
| Mask support | No | Yes |
| Video support | No | Yes |
| Native tensor ops | Partial | Full |
| CutMix/MixUp | Manual | Built-in |
| AutoAugment | Limited | Full |
| `ToTensor` | Yes (deprecated in v2) | Use `ToImage` + `ToDtype` |
