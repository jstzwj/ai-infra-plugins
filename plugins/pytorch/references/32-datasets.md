# PyTorch Datasets - Comprehensive Reference

This chapter covers all dataset classes and utilities in `torch.utils.data`, including map-style and iterable-style datasets, built-in dataset types, dataset composition, splitting, custom dataset patterns, and distributed training support.

---

## 1. torch.utils.data.Dataset (Map-Style)

The abstract base class for all map-style datasets. A map-style dataset implements `__getitem__()` and `__len__()` protocols, representing a map from (possibly non-integral) indices to data samples.

### Definition

```python
class torch.utils.data.Dataset:
    """
    An abstract class representing a Dataset.

    All datasets that represent a map from keys to data samples should
    subclass it. All subclasses should override __getitem__(), supporting
    fetching a data sample for a given key. Subclasses should also
    override __len__(), returning the size of the dataset.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])
```

### Key Methods

#### __getitem__(self, index)
Fetches the data sample at the given index. The index can be an integer or any hashable type.

```python
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.transform = transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        label = self._get_label(self.image_files[idx])

        if self.transform:
            image = self.transform(image)

        return image, label

    def _get_label(self, filename):
        # Parse label from filename
        return int(filename.split('_')[0])
```

#### __len__(self)
Returns the total number of samples in the dataset.

```python
class ImageDataset(Dataset):
    def __len__(self):
        return len(self.image_files)
```

### Subclassing Patterns

#### Basic Image Classification Dataset

```python
import os
from PIL import Image
from torch.utils.data import Dataset

class ImageClassificationDataset(Dataset):
    """
    Generic image classification dataset.
    Expected directory structure:
        root/
            class_0/
                img001.jpg
                img002.jpg
            class_1/
                img003.jpg
                img004.jpg
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                samples.append((img_path, self.class_to_idx[cls_name]))
        return samples

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.samples)
```

#### Image + Metadata Dataset

```python
class ImageMetadataDataset(Dataset):
    """Dataset returning image, label, and metadata dict."""

    def __init__(self, annotations, image_dir, transform=None):
        self.annotations = annotations  # List of dicts
        self.image_dir = image_dir
        self.transform = transform

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.image_dir, ann['filename'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': ann['label'],
            'bbox': torch.tensor(ann['bbox'], dtype=torch.float32),
            'metadata': {
                'filename': ann['filename'],
                'split': ann.get('split', 'train'),
            }
        }

    def __len__(self):
        return len(self.annotations)
```

---

## 2. torch.utils.data.IterableDataset

An iterable-style dataset that implements `__iter__()`. Useful for streaming data, infinite data sources, or when random access is expensive.

### Definition

```python
class torch.utils.data.IterableDataset:
    """
    An iterable Dataset.

    All datasets that represent an iterable of data samples should subclass it.
    Such form of datasets is particularly useful when data come from a stream.

    All subclasses should implement __iter__(), yielding data samples.
    """

    def __iter__(self):
        raise NotImplementedError
```

### Key Methods

#### __iter__(self)
Returns an iterator over the dataset samples. Called each time the DataLoader iterates over the dataset.

```python
class StreamDataset(IterableDataset):
    def __init__(self, data_stream):
        self.data_stream = data_stream

    def __iter__(self):
        for item in self.data_stream:
            yield self._process(item)

    def _process(self, item):
        return torch.tensor(item['features']), item['label']
```

### Multi-Worker IterableDataset

```python
import math
import torch
from torch.utils.data import IterableDataset, get_worker_info

class MultiWorkerIterableDataset(IterableDataset):
    """Properly handles multi-worker data loading."""

    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # Single-process loading
            iter_start = 0
            iter_end = len(self.data_list)
        else:
            # Multi-process loading: partition data
            per_worker = int(math.ceil(
                len(self.data_list) / float(worker_info.num_workers)
            ))
            iter_start = worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, len(self.data_list))

        for i in range(iter_start, iter_end):
            yield self._process_item(self.data_list[i])

    def _process_item(self, item):
        return torch.tensor(item['data']), item['label']
```

### Distributed IterableDataset

```python
class DistributedIterableDataset(IterableDataset):
    """IterableDataset that works with DistributedDataParallel."""

    def __init__(self, data_source, world_size, rank):
        super().__init__()
        self.data_source = data_source
        self.world_size = world_size
        self.rank = rank

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        total_workers = self.world_size * num_workers
        global_worker_id = self.rank * num_workers + worker_id

        for i in range(global_worker_id, len(self.data_source), total_workers):
            yield self.data_source[i]
```

### Infinite IterableDataset

```python
import itertools

class InfiniteDataset(IterableDataset):
    """An infinite dataset that wraps a finite dataset."""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        while True:
            for item in self.dataset:
                yield item
```

---

## 3. TensorDataset

A dataset wrapping tensors. Each sample is retrieved by indexing tensors along the first dimension.

### Constructor

```python
torch.utils.data.TensorDataset(*tensors)
```

**Parameters:**
- `*tensors` (Tensor): Tensors that have the same size of the first dimension.

### Usage

```python
from torch.utils.data import TensorDataset

# Create from raw tensors
features = torch.randn(1000, 10)    # 1000 samples, 10 features
labels = torch.randint(0, 2, (1000,))  # Binary labels

dataset = TensorDataset(features, labels)

# Access individual samples
features_sample, labels_sample = dataset[0]
print(features_sample.shape)  # torch.Size([10])
print(labels_sample.shape)    # torch.Size([])

# Use with DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch_features, batch_labels in loader:
    print(batch_features.shape)  # torch.Size([32, 10])
    print(batch_labels.shape)    # torch.Size([32])
```

### Multi-Tensor Dataset

```python
# More than two tensors
images = torch.randn(1000, 3, 224, 224)
labels = torch.randint(0, 10, (1000,))
bbox = torch.randn(1000, 4)
masks = torch.randint(0, 2, (1000, 224, 224))

dataset = TensorDataset(images, labels, bbox, masks)

# Access all four tensors per sample
img, lbl, box, mask = dataset[42]
```

### TensorDataset with Transforms

```python
class TransformTensorDataset(Dataset):
    """TensorDataset with on-the-fly transforms."""

    def __init__(self, *tensors, transform=None):
        assert all(t.size(0) == tensors[0].size(0) for t in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, idx):
        items = tuple(t[idx] for t in self.tensors)
        if self.transform:
            items = self.transform(*items)
        return items

    def __len__(self):
        return self.tensors[0].size(0)
```

---

## 4. ConcatDataset

A dataset that concatenates multiple datasets. All datasets must be map-style.

### Constructor

```python
torch.utils.data.ConcatDataset(datasets)
```

**Parameters:**
- `datasets` (sequence): List of datasets to be concatenated.

### Usage

```python
from torch.utils.data import ConcatDataset, TensorDataset

# Create two separate datasets
data1 = TensorDataset(torch.randn(100, 10), torch.randint(0, 5, (100,)))
data2 = TensorDataset(torch.randn(200, 10), torch.randint(0, 5, (200,)))

# Concatenate them
combined = ConcatDataset([data1, data2])

print(len(combined))  # 300

# Access samples: indices 0-99 from data1, 100-299 from data2
sample = combined[150]  # From data2 (index 50 within data2)
```

### Attributes and Methods

```python
combined = ConcatDataset([data1, data2])

# Attributes
print(len(combined))            # Total number of samples
print(combined.datasets)        # List of constituent datasets
print(combined.cumulative_sizes) # Cumulative sizes [100, 300]

# Methods
combined.get_dataset(idx)       # Returns (dataset_index, sample_index_within_dataset)
```

### Practical ConcatDataset Example

```python
# Combine datasets from multiple sources
train_datasets = []
for city in ['nyc', 'la', 'chicago']:
    ds = CityDataset(
        data_dir=f'/data/{city}',
        split='train',
        transform=train_transform,
    )
    train_datasets.append(ds)

train_dataset = ConcatDataset(train_datasets)
print(f"Total training samples: {len(train_dataset)}")
```

---

## 5. ChainDataset

A dataset that chains multiple IterableDatasets. Unlike ConcatDataset, this is for iterable-style datasets.

### Constructor

```python
torch.utils.data.ChainDataset(datasets)
```

**Parameters:**
- `datasets` (iterable of IterableDataset): Datasets to chain.

### Usage

```python
from torch.utils.data import ChainDataset, IterableDataset

class RangeDataset(IterableDataset):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __iter__(self):
        for i in range(self.start, self.end):
            yield i

ds1 = RangeDataset(0, 5)   # Yields: 0, 1, 2, 3, 4
ds2 = RangeDataset(10, 15) # Yields: 10, 11, 12, 13, 14

chained = ChainDataset([ds1, ds2])

# Iterating yields: 0, 1, 2, 3, 4, 10, 11, 12, 13, 14
for item in chained:
    print(item)
```

### ChainDataset vs ConcatDataset

| Feature | ConcatDataset | ChainDataset |
|---------|---------------|--------------|
| Dataset type | Map-style (`Dataset`) | Iterable-style (`IterableDataset`) |
| Access pattern | Random by index | Sequential iteration |
| `__len__` | Yes (sum of sizes) | No (infinite streams possible) |
| `__getitem__` | Yes | No |
| Use case | Finite, indexable data | Streaming data |

---

## 6. Subset

A subset of a dataset at specified indices.

### Constructor

```python
torch.utils.data.Subset(dataset, indices)
```

**Parameters:**
- `dataset` (Dataset): The whole dataset.
- `indices` (sequence): Indices into the dataset.

### Usage

```python
from torch.utils.data import Subset

# Create a subset with specific indices
full_dataset = TensorDataset(torch.randn(1000, 10), torch.randint(0, 5, (1000,)))

# Select first 100 samples
subset = Subset(full_dataset, range(100))
print(len(subset))  # 100

# Access samples
data, label = subset[0]  # Same as full_dataset[0]

# Use indices to create arbitrary subsets
even_indices = list(range(0, 1000, 2))
even_subset = Subset(full_dataset, even_indices)
print(len(even_subset))  # 500
```

### Subset Attributes

```python
subset = Subset(full_dataset, [0, 5, 10, 15])

print(subset.dataset)    # The underlying dataset
print(subset.indices)    # The indices tensor/list
print(len(subset))       # len(indices)
```

### Train/Val/Test Split with Subset

```python
import numpy as np
from torch.utils.data import Subset

def train_val_test_split(dataset, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split a dataset into train/val/test subsets."""
    n = len(dataset)
    indices = list(range(n))
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_subset = Subset(dataset, indices[:train_end])
    val_subset = Subset(dataset, indices[train_end:val_end])
    test_subset = Subset(dataset, indices[val_end:])

    return train_subset, val_subset, test_subset

train_data, val_data, test_data = train_val_test_split(full_dataset)
print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
```

---

## 7. random_split

Randomly split a dataset into non-overlapping new datasets of given lengths.

### Function Signature

```python
torch.utils.data.random_split(
    dataset,
    lengths,
    generator=<default generator>,
)
```

**Parameters:**
- `dataset` (Dataset): Dataset to be split.
- `lengths` (sequence): Lengths of splits to be produced. If sum of lengths is less than dataset size, remaining data is ignored.
- `generator` (torch.Generator, optional): Generator used for the random permutation.

### Usage

```python
from torch.utils.data import random_split

dataset = TensorDataset(torch.randn(1000, 10), torch.randint(0, 5, (1000,)))

# Split into 70% train, 15% val, 15% test
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [700, 150, 150]
)

print(len(train_dataset))  # 700
print(len(val_dataset))    # 150
print(len(test_dataset))   # 150

# Each returned value is a Subset object
assert isinstance(train_dataset, torch.utils.data.Subset)
```

### Reproducible Splitting

```python
g = torch.Generator()
g.manual_seed(42)

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [700, 150, 150], generator=g
)
```

### Percentage-Based Splitting

```python
def random_split_pct(dataset, train_pct=0.8, val_pct=0.1, test_pct=0.1, seed=42):
    """Split dataset by percentage."""
    total = len(dataset)
    train_len = int(total * train_pct)
    val_len = int(total * val_pct)
    test_len = total - train_len - val_len  # Ensure sum equals total

    g = torch.Generator()
    g.manual_seed(seed)

    return random_split(dataset, [train_len, val_len, test_len], generator=g)
```

### Preserving Split Consistency

```python
import os

def get_or_create_split(dataset, split_path, lengths, seed=42):
    """Load split indices from file, or create and save them."""
    if os.path.exists(split_path):
        indices = torch.load(split_path)
    else:
        g = torch.Generator()
        g.manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=g).tolist()
        torch.save(indices, split_path)

    splits = []
    offset = 0
    for length in lengths:
        splits.append(Subset(dataset, indices[offset:offset + length]))
        offset += length

    return splits
```

---

## 8. Custom Dataset Examples

### Image Folder Dataset with Caching

```python
import os
import pickle
from PIL import Image
from torch.utils.data import Dataset

class CachedImageDataset(Dataset):
    """Image dataset with optional disk caching for decoded images."""

    def __init__(self, root_dir, transform=None, cache_dir=None):
        self.root_dir = root_dir
        self.transform = transform
        self.cache_dir = cache_dir

        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                self.samples.append((
                    os.path.join(cls_dir, fname),
                    self.class_to_idx[cls]
                ))

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Try loading from cache
        if self.cache_dir:
            cache_path = os.path.join(
                self.cache_dir, f'{idx}.pkl'
            )
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    image = pickle.load(f)
            else:
                image = Image.open(img_path).convert('RGB')
                os.makedirs(self.cache_dir, exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump(image, f)
        else:
            image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.samples)
```

### HDF5 Dataset

```python
import h5py
from torch.utils.data import Dataset

class HDF5Dataset(Dataset):
    """Load data from HDF5 files efficiently."""

    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform

        # Get length without keeping file open
        with h5py.File(h5_path, 'r') as f:
            self.length = f['images'].shape[0]

    def __getitem__(self, idx):
        # Open file for each access (thread-safe for multi-worker loading)
        with h5py.File(self.h5_path, 'r') as f:
            image = f['images'][idx]
            label = f['labels'][idx]

        image = torch.from_numpy(image).float()
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.length
```

### LMDB Dataset

```python
import lmdb
import pickle
from torch.utils.data import Dataset

class LMDBDataset(Dataset):
    """Load data from LMDB database."""

    def __init__(self, lmdb_path, transform=None):
        self.lmdb_path = lmdb_path
        self.transform = transform

        # Get length
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            self.length = txn.stat()['entries']
        env.close()

    def __getitem__(self, idx):
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            data = txn.get(str(idx).encode())
            sample = pickle.loads(data)
        env.close()

        image = sample['image']
        label = sample['label']

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.length
```

### Multi-Label Classification Dataset

```python
class MultiLabelDataset(Dataset):
    """Dataset for multi-label classification tasks."""

    def __init__(self, image_dir, annotations, transform=None, num_classes=20):
        self.image_dir = image_dir
        self.annotations = annotations  # List of dicts with 'filename' and 'labels'
        self.transform = transform
        self.num_classes = num_classes

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.image_dir, ann['filename'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Multi-hot encoding
        labels = torch.zeros(self.num_classes, dtype=torch.float32)
        for label_idx in ann['labels']:
            labels[label_idx] = 1.0

        return image, labels

    def __len__(self):
        return len(self.annotations)
```

### Video Dataset

```python
class VideoDataset(Dataset):
    """Dataset for video data, loading sequences of frames."""

    def __init__(self, video_dir, annotations, num_frames=16, transform=None):
        self.video_dir = video_dir
        self.annotations = annotations
        self.num_frames = num_frames
        self.transform = transform

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        video_path = os.path.join(self.video_dir, ann['video_id'])
        frames = self._load_frames(video_path, self.num_frames)

        if self.transform:
            frames = torch.stack([self.transform(f) for f in frames])

        return frames, ann['label']

    def _load_frames(self, video_path, num_frames):
        """Load uniformly sampled frames from video."""
        import cv2
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        return frames

    def __len__(self):
        return len(self.annotations)
```

### Audio Dataset

```python
import torchaudio

class AudioDataset(Dataset):
    """Dataset for audio classification."""

    def __init__(self, audio_dir, annotations, sample_rate=16000,
                 max_length=5.0, transform=None):
        self.audio_dir = audio_dir
        self.annotations = annotations
        self.sample_rate = sample_rate
        self.max_samples = int(max_length * sample_rate)
        self.transform = transform

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        audio_path = os.path.join(self.audio_dir, ann['filename'])

        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Pad or truncate to fixed length
        if waveform.size(1) > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        elif waveform.size(1) < self.max_samples:
            padding = self.max_samples - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, ann['label']

    def __len__(self):
        return len(self.annotations)
```

---

## 9. Map-Style vs Iterable-Style Comparison

### Detailed Comparison

| Aspect | Map-Style (Dataset) | Iterable-Style (IterableDataset) |
|--------|---------------------|----------------------------------|
| Required methods | `__getitem__`, `__len__` | `__iter__` |
| Index access | Yes (`dataset[idx]`) | No |
| Shuffling | Via `shuffle=True` or `Sampler` | Must implement internally |
| Data source | Files, in-memory | Streams, generators |
| Caching | Easy (index-based) | Harder |
| Splitting | `Subset`, `random_split` | Manual sharding |
| Distributed | `DistributedSampler` | Manual rank-based sharding |
| Batching | DataLoader handles it | DataLoader handles it |
| `__len__` | Required | Optional |
| Random access cost | O(1) to O(n) | Must iterate to reach point |

### When to Use Which

```python
# Use Map-Style when:
# - Data fits in memory or can be randomly accessed (files on disk)
# - You need random access by index
# - You want to use Samplers for custom sampling strategies
# - You want to split the dataset easily

class MapStyleUseCase(Dataset):
    """Good for: images on disk, tabular data, etc."""
    def __init__(self, file_list):
        self.file_list = file_list

    def __getitem__(self, idx):
        return self._load(self.file_list[idx])

    def __len__(self):
        return len(self.file_list)


# Use Iterable-Style when:
# - Data comes from a stream (network, pipe)
# - Data is generated on-the-fly
# - Random access is very expensive
# - Working with infinite data sources

class IterableStyleUseCase(IterableDataset):
    """Good for: data streams, generators, remote data."""
    def __init__(self, stream_url):
        self.stream_url = stream_url

    def __iter__(self):
        for item in self._stream(self.stream_url):
            yield item
```

---

## 10. Dataset Utilities

### get_worker_info

```python
torch.utils.data.get_worker_info()
```

Returns information about the current DataLoader worker. Returns `None` in the main process.

```python
import torch.utils.data as data_utils

def my_worker_init(worker_id):
    info = data_utils.get_worker_info()
    if info is not None:
        print(f"Worker ID: {info.id}")
        print(f"Num workers: {info.num_workers}")
        print(f"Seed: {info.seed}")
        print(f"Dataset: {info.dataset}")
```

### default_collate

```python
torch.utils.data.default_collate(batch)
```

The default collate function used by DataLoader. Handles:
- `torch.Tensor` -> stacks along dim 0
- `numpy.ndarray` -> converts to tensor, then stacks
- `float`/`int` -> converts to tensor
- `str` -> keeps as list
- `Mapping` -> applies collate to each value
- `Sequence` -> applies collate to each element

```python
from torch.utils.data.dataloader import default_collate

# Manually collate a list of samples
samples = [
    (torch.tensor([1.0, 2.0]), 0),
    (torch.tensor([3.0, 4.0]), 1),
]
batch = default_collate(samples)
# batch[0]: tensor([[1., 2.], [3., 4.]])
# batch[1]: tensor([0, 1])
```

### default_convert

```python
torch.utils.data.default_convert(data)
```

Converts each data element into a tensor. Unlike `default_collate`, does not batch.

```python
from torch.utils.data.dataloader import default_convert

# Convert numpy array to tensor
import numpy as np
arr = np.array([1.0, 2.0, 3.0])
tensor = default_convert(arr)  # tensor([1., 2., 3.])
```

### Dataset Registration Pattern

```python
# Registry pattern for dataset management
DATASET_REGISTRY = {}

def register_dataset(name):
    """Decorator to register datasets."""
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator

@register_dataset('classification')
class ClassificationDataset(Dataset):
    def __init__(self, root, split, transform=None):
        # ...
        pass

@register_dataset('detection')
class DetectionDataset(Dataset):
    def __init__(self, root, split, transform=None):
        # ...
        pass

def build_dataset(name, **kwargs):
    """Build a dataset by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. "
                        f"Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name](**kwargs)
```

---

## 11. Fault-Tolerant Dataset Patterns

### Error-Handling Dataset

```python
class SafeDataset(Dataset):
    """Dataset that handles errors in __getitem__ gracefully."""

    def __init__(self, dataset, cache_valid_indices=True):
        self._dataset = dataset
        self._cache_valid_indices = cache_valid_indices
        self._valid_indices = None

        if cache_valid_indices:
            self._build_valid_index_map()

    def _build_valid_index_map(self):
        """Pre-validate all indices."""
        self._valid_indices = []
        for i in range(len(self._dataset)):
            try:
                _ = self._dataset[i]
                self._valid_indices.append(i)
            except Exception as e:
                print(f"Warning: Skipping index {i}: {e}")

    def __getitem__(self, idx):
        if self._valid_indices is not None:
            actual_idx = self._valid_indices[idx]
        else:
            actual_idx = idx
        return self._dataset[actual_idx]

    def __len__(self):
        if self._valid_indices is not None:
            return len(self._valid_indices)
        return len(self._dataset)
```

### Lazy Loading Dataset

```python
class LazyDataset(Dataset):
    """Dataset with lazy loading - files opened only when accessed."""

    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self._cache = {}  # Optional cache

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        data = self._load(self.file_paths[idx])
        if self.transform:
            data = self.transform(data)

        result = (data, self.labels[idx])

        # Optionally cache
        if len(self._cache) < 1000:
            self._cache[idx] = result

        return result

    def __len__(self):
        return len(self.file_paths)

    def _load(self, path):
        # Load data from file
        pass
```

---

## 12. Complete Example: End-to-End Dataset Pipeline

```python
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from typing import Tuple, Optional, Callable

class CustomImageDataset(Dataset):
    """
    A complete, production-ready image classification dataset.
    Supports caching, transforms, and error handling.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        cache_size: int = 0,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.cache_size = cache_size
        self._cache = {}

        # Discover classes and samples
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for fname in sorted(os.listdir(cls_dir)):
                self.samples.append((
                    os.path.join(cls_dir, fname),
                    self.class_to_idx[cls_name],
                ))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx in self._cache:
            return self._cache[idx]

        img_path, label = self.samples[idx]

        try:
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading {img_path}: {e}") from e

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        result = (image, label)

        if self.cache_size > 0 and len(self._cache) < self.cache_size:
            self._cache[idx] = result

        return result

    def __len__(self) -> int:
        return len(self.samples)

    def get_class_distribution(self) -> dict:
        """Return the number of samples per class."""
        distribution = {c: 0 for c in self.classes}
        for _, label in self.samples:
            distribution[self.classes[label]] += 1
        return distribution


# Usage
if __name__ == '__main__':
    from torchvision import transforms

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
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

    full_dataset = CustomImageDataset(
        root_dir='/path/to/data',
        transform=train_transform,
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}, Labels: {labels.shape}")
        break
```
