# PyTorch DataLoader - Comprehensive Reference

This chapter covers `torch.utils.data.DataLoader` and all related components for efficient data loading in PyTorch. The DataLoader is the core utility for feeding data into training and inference loops, providing batching, shuffling, multi-process loading, automatic collation, and more.

---

## 1. torch.utils.data.DataLoader

### Constructor

```python
torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=None,
    sampler=None,
    batch_sampler=None,
    num_workers=0,
    collate_fn=None,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
    multiprocessing_context=None,
    generator=None,
    *,
    prefetch_factor=None,
    persistent_workers=False,
    pin_memory_device="",
)
```

### Parameters (Detailed)

#### dataset (Dataset or IterableDataset)
The dataset object from which to load data. Must be either:
- A **map-style dataset** (subclass of `Dataset`) that implements `__getitem__()` and `__len__()`.
- An **iterable-style dataset** (subclass of `IterableDataset`) that implements `__iter__()`.

```python
from torch.utils.data import DataLoader, Dataset, IterableDataset

# Map-style dataset
class MyMapDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)

# Iterable-style dataset
class MyIterableDataset(IterableDataset):
    def __init__(self, start, end):
        super().__init__()
        self.start = start
        self.end = end

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = self.start
            iter_end = self.end
        else:
            per_worker = int(
                math.ceil((self.end - self.start) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(range(iter_start, iter_end))
```

#### batch_size (int, optional)
Number of samples per batch. Default: `1`.

```python
# Batch of 32 samples
loader = DataLoader(dataset, batch_size=32)

# Iterate over batches
for batch_data, batch_labels in loader:
    print(batch_data.shape)  # torch.Size([32, ...])
```

#### shuffle (bool, optional)
Set to `True` to have the data reshuffled at every epoch. Default: `False`. This argument is mutually exclusive with `sampler` and `batch_sampler`.

```python
# Shuffle training data
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Do NOT shuffle validation/test data
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
```

**Note:** IterableDataset does not support `shuffle=True`. Use the dataset's internal shuffling mechanism instead.

#### sampler (Sampler or Iterable, optional)
Defines the strategy to draw samples from the dataset. If specified, `shuffle` must be `False`.

```python
from torch.utils.data import WeightedRandomSampler

# Handle class imbalance with weighted sampling
class_counts = [100, 500, 200]  # samples per class
weights = [1.0 / c for c in class_counts]
sample_weights = [weights[label] for _, label in dataset]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True
)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

#### batch_sampler (Sampler or Iterable, optional)
Like `sampler`, but returns a batch of indices at a time. Mutually exclusive with `batch_size`, `shuffle`, `sampler`, and `drop_last`.

```python
from torch.utils.data import BatchSampler, SequentialSampler

# Create batches manually
batch_sampler = BatchSampler(
    SequentialSampler(dataset),
    batch_size=32,
    drop_last=False
)
loader = DataLoader(dataset, batch_sampler=batch_sampler)
```

#### num_workers (int, optional)
Number of subprocesses to use for data loading. `0` means data is loaded in the main process.

```python
# Single-process loading (default, good for debugging)
loader = DataLoader(dataset, num_workers=0)

# Multi-process loading for better throughput
loader = DataLoader(dataset, num_workers=4)

# Common rule of thumb: 4 * number of GPUs
loader = DataLoader(dataset, num_workers=8)
```

**Performance considerations:**
- More workers increases memory usage
- Too many workers can cause overhead from process spawning
- On Windows, use `if __name__ == '__main__':` guard
- On Linux, `fork` is default; on Windows/macOS, `spawn` is used

#### collate_fn (callable, optional)
Merges a list of samples to form a mini-batch. Default collate handles tensors, numpy arrays, numbers, strings, dicts, and lists.

```python
# Default collate behavior
# Input:  [tensor1, tensor2, tensor3, ...]
# Output: batched_tensor (stacked along dim 0)

# Custom collate for variable-length sequences
def custom_collate_fn(batch):
    """
    Args:
        batch: List of (sequence, label) tuples
    Returns:
        padded_sequences, lengths, labels
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    # Pad sequences to max length in batch
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    labels = torch.tensor(labels)
    return padded, lengths, labels

loader = DataLoader(dataset, batch_size=32, collate_fn=custom_collate_fn)

# Custom collate for dictionaries
def dict_collate_fn(batch):
    """Handle batch of dicts with varying keys."""
    return {
        key: torch.stack([d[key] for d in batch])
        for key in batch[0]
    }
```

#### pin_memory (bool, optional)
If `True`, the DataLoader will copy tensors into CUDA pinned memory before returning them. Default: `False`.

```python
# Enable pinned memory for faster CPU-to-GPU transfer
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    pin_memory=True
)

# Works with automatic device transfer
for data, target in loader:
    # data is in pinned memory, .to('cuda') is faster
    data = data.to('cuda', non_blocking=True)
    target = target.to('cuda', non_blocking=True)
```

**When to use pin_memory:**
- Always recommended when training on GPU
- Provides ~2x speedup for CPU-to-GPU transfers
- Uses page-locked (pinned) host memory
- Only affects CPU tensors being transferred to CUDA

#### drop_last (bool, optional)
Set to `True` to drop the last incomplete batch. Default: `False`.

```python
# Drop last batch to maintain consistent batch size (e.g., for BatchNorm)
loader = DataLoader(dataset, batch_size=64, drop_last=True)
```

#### timeout (numeric, optional)
If positive, the timeout value for collecting a batch from workers. Default: `0` (no timeout). Should always be non-negative.

```python
# Set a 30-second timeout for batch collection
loader = DataLoader(dataset, batch_size=64, num_workers=4, timeout=30)
```

#### worker_init_fn (callable, optional)
If not `None`, this is called on each worker subprocess with the worker id (`int` in `[0, num_workers - 1]`) as input, after seeding and before data loading.

```python
import numpy as np
import random

def seed_worker(worker_id):
    """Initialize worker with proper seeding for reproducibility."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=g,
)
```

#### multiprocessing_context (str or multiprocessing.Context, optional)
The multiprocessing context for worker process creation. Can be `'fork'`, `'spawn'`, or `'forkserver'`.

```python
# Use forkserver for safer multiprocessing
loader = DataLoader(
    dataset,
    num_workers=4,
    multiprocessing_context='forkserver'
)

# Use spawn (required for CUDA in workers on some systems)
loader = DataLoader(
    dataset,
    num_workers=4,
    multiprocessing_context='spawn'
)
```

#### generator (torch.Generator, optional)
A generator used by the RandomSampler to generate random indexes, and by the default `collate_fn` to generate random stacks.

```python
# Reproducible data loading
g = torch.Generator()
g.manual_seed(0)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    generator=g,
)
```

#### prefetch_factor (int, optional, keyword-only)
Number of batches loaded in advance by each worker. `2` means there will be a total of 2 * num_workers batches prefetched across all workers. Default value depends on PyTorch version.

```python
# Prefetch 4 batches per worker
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    prefetch_factor=4,
)
```

#### persistent_workers (bool, optional, keyword-only)
If `True`, the data loader will not shut down the worker processes after a dataset has been consumed once. Default: `False`.

```python
# Keep workers alive across epochs (avoids restart overhead)
loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
)
```

**Benefits of persistent_workers:**
- Avoids per-epoch worker startup cost
- Particularly useful for small datasets where startup dominates
- Increases memory usage since workers stay alive
- Requires `num_workers > 0`

---

## 2. Multi-Process Loading

### How Multi-Process Loading Works

When `num_workers > 0`, the DataLoader spawns worker processes that preload batches in parallel with training.

```
Main Process                Worker 0    Worker 1    Worker 2    Worker 3
    |                          |           |           |           |
    |-- request batch -------->|           |           |           |
    |                          |-- load -->|           |           |
    |<-- return batch ---------|           |           |           |
    |-- request batch -------->|           |-- load -->|           |
    |                          |           |           |-- load -->|
    |<-- return batch ---------|           |<----------|           |
    ...
```

### Worker Lifecycle

```python
# Workers are initialized at first iteration
loader = DataLoader(dataset, num_workers=4)

# Workers spawned here (first __iter__ call)
for i, batch in enumerate(loader):
    # Workers are alive during iteration
    pass

# Workers are shut down after iteration completes
# (unless persistent_workers=True)
```

### Worker Information API

```python
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    print(f"Worker {worker_id}:")
    print(f"  num_workers: {worker_info.num_workers}")
    print(f"  seed: {worker_info.seed}")
    print(f"  dataset: {worker_info.dataset}")

loader = DataLoader(
    dataset,
    num_workers=4,
    worker_init_fn=worker_init_fn,
)
```

### Shard-Based Data Loading with Multiple Workers

```python
class ShardedIterableDataset(torch.utils.data.IterableDataset):
    """Split data across workers for iterable-style datasets."""

    def __init__(self, data_source):
        super().__init__()
        self.data_source = data_source

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process: yield all data
            yield from self.data_source
        else:
            # Multi-process: shard the data
            per_worker = int(math.ceil(
                len(self.data_source) / float(worker_info.num_workers)
            ))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.data_source))
            yield from self.data_source[start:end]

dataset = ShardedIterableDataset(range(1000))
loader = DataLoader(dataset, batch_size=32, num_workers=4)
```

### Memory Considerations

```python
import resource

# Check memory limits
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
print(f"File descriptor limit: soft={soft}, hard={hard}")

# Increase if needed (for many workers with many file handles)
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

# Each worker copies the dataset object, so be mindful of dataset size
# For large in-memory datasets, use shared memory or memory-mapped files
```

---

## 3. Automatic Batching vs Manual Batching

### Automatic Batching (batch_size != None)

When `batch_size` is set (default=1), the DataLoader automatically collates individual samples into batches.

```python
# Automatic batching
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_x, batch_y in loader:
    # batch_x: torch.Size([32, ...]) - automatically stacked
    # batch_y: torch.Size([32]) - automatically stacked
    pass
```

### Manual Batching (batch_size = None)

When `batch_size=None`, the DataLoader passes individual samples through without any automatic batching. Useful when the dataset already returns batches.

```python
class PreBatchedDataset(Dataset):
    """Dataset where each item is already a batch."""
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = len(data) // batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        return self.data[start:end], self.labels[start:end]

    def __len__(self):
        return self.num_batches

# Manual batching: each item from dataset IS a batch
loader = DataLoader(prebatched_dataset, batch_size=None)

for batch_x, batch_y in loader:
    # batch_x already has shape [batch_size, ...]
    pass
```

### Comparison Table

| Feature | Automatic Batching | Manual Batching |
|---------|-------------------|-----------------|
| `batch_size` | Set to integer | `None` |
| `collate_fn` | Applied | Not applied |
| `drop_last` | Applied | Not applicable |
| `sampler` | Applied | Not applicable |
| `shuffle` | Applied | Not applicable |
| Data shape | Individual samples stacked | As-is from dataset |

---

## 4. collate_fn: Default and Custom

### Default collate_fn

The default collate function handles:
- `torch.Tensor` -> stacked along dim 0
- `numpy.ndarray` -> converted to tensor, then stacked
- `float`, `int` -> converted to tensor, then stacked
- `str` -> list of strings
- `dict` -> dict with batched values
- `list` / `tuple` -> recursively collated

```python
# Example: default collate with dict output
class DictDataset(Dataset):
    def __getitem__(self, idx):
        return {
            'image': torch.randn(3, 224, 224),
            'label': idx % 10,
            'metadata': {'id': idx, 'source': 'train'}
        }

    def __len__(self):
        return 100

loader = DataLoader(DictDataset(), batch_size=4)
batch = next(iter(loader))
# batch['image'].shape == torch.Size([4, 3, 224, 224])
# batch['label'].shape == torch.Size([4])
# batch['metadata'] == {'id': [0,1,2,3], 'source': ['train']*4}
```

### Custom collate_fn Examples

#### Variable-Length Sequence Collation

```python
from torch.nn.utils.rnn import pad_sequence

def pad_collate_fn(batch):
    """
    Collate function for variable-length sequences.
    Pads sequences to the length of the longest in the batch.
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])

    # Sort by length (descending) for pack_padded_sequence
    lengths, sort_idx = lengths.sort(descending=True)
    sorted_seqs = [sequences[i] for i in sort_idx]

    padded_seqs = pad_sequence(sorted_seqs, batch_first=True)
    labels = torch.tensor(labels)[sort_idx]

    return padded_seqs, lengths, labels
```

#### Nested Structure Collation

```python
def nested_collate_fn(batch):
    """Handle complex nested data structures."""
    def collate_recursive(items):
        if isinstance(items[0], torch.Tensor):
            return torch.stack(items)
        elif isinstance(items[0], dict):
            return {
                key: collate_recursive([item[key] for item in items])
                for key in items[0]
            }
        elif isinstance(items[0], (list, tuple)):
            return type(items[0])(
                collate_recursive([item[i] for item in items])
                for i in range(len(items[0]))
            )
        else:
            return items

    return collate_recursive(batch)
```

#### Mixed-Type Collation

```python
def mixed_collate_fn(batch):
    """Handle batches with mixed types (images, text, labels)."""
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    bbox = torch.stack([item['bbox'] for item in batch])

    return {
        'images': images,
        'texts': texts,
        'labels': labels,
        'bbox': bbox,
    }
```

---

## 5. worker_init_fn for Seeding

### Why Seeding Matters

Each worker process has its own Python and NumPy random state. Without proper seeding, workers may produce identical or predictable random numbers, affecting data augmentation reproducibility.

### Basic Seeding Pattern

```python
import numpy as np
import random

def seed_worker(worker_id):
    """Properly seed each worker for reproducibility."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=g,
)
```

### Per-Epoch Seeding

```python
def get_worker_init_fn(epoch):
    """Create a worker_init_fn that varies by epoch."""
    def worker_init_fn(worker_id):
        # Combine epoch and worker_id for unique seed
        seed = 42 + epoch * 1000 + worker_id
        np.random.seed(seed)
        random.seed(seed)
    return worker_init_fn

# Use different seeds per epoch
for epoch in range(num_epochs):
    loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        worker_init_fn=get_worker_init_fn(epoch),
        shuffle=True,
    )
    for batch in loader:
        train_step(batch)
```

### Controlling All Sources of Randomness

```python
import os

def full_seed_worker(worker_id):
    """Complete seeding including all random sources."""
    seed = torch.initial_seed() % (2**32)

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (within worker)
    torch.manual_seed(seed)

    # Environment variable (for any library that uses it)
    os.environ['PYTHONHASHSEED'] = str(seed)
```

---

## 6. IterableDataset Patterns

### Basic IterableDataset

```python
class StreamDataset(torch.utils.data.IterableDataset):
    """Stream data from a source (e.g., file, network)."""

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path, 'r') as f:
            for line in f:
                data = self._parse_line(line)
                yield data

    def _parse_line(self, line):
        values = list(map(float, line.strip().split(',')))
        return torch.tensor(values[:-1]), int(values[-1])
```

### Distributed IterableDataset

```python
class DistributedIterableDataset(torch.utils.data.IterableDataset):
    """IterableDataset for distributed training."""

    def __init__(self, data_source, world_size, rank):
        super().__init__()
        self.data_source = data_source
        self.world_size = world_size
        self.rank = rank

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Split across workers within a rank
            per_worker = len(self.data_source) // (
                self.world_size * worker_info.num_workers
            )
            worker_global_id = self.rank * worker_info.num_workers + worker_info.id
            start = worker_global_id * per_worker
            end = start + per_worker
        else:
            per_worker = len(self.data_source) // self.world_size
            start = self.rank * per_worker
            end = start + per_worker

        for i in range(start, end):
            yield self.data_source[i]
```

### Generator-Based IterableDataset

```python
class GeneratorIterableDataset(torch.utils.data.IterableDataset):
    """Generate data on-the-fly using a generator function."""

    def __init__(self, generator_fn, length):
        super().__init__()
        self.generator_fn = generator_fn
        self.length = length

    def __iter__(self):
        return self.generator_fn()

    def __len__(self):
        return self.length

def data_generator():
    """Infinite or finite data generator."""
    for i in range(10000):
        x = torch.randn(10)
        y = (x.sum() > 0).long()
        yield x, y
```

### IterableDataset with Shuffling

```python
class ShuffledIterableDataset(torch.utils.data.IterableDataset):
    """IterableDataset with buffer-based shuffling."""

    def __init__(self, data_source, buffer_size=1000):
        super().__init__()
        self.data_source = data_source
        self.buffer_size = buffer_size

    def __iter__(self):
        buffer = []
        for item in self.data_source:
            buffer.append(item)
            if len(buffer) >= self.buffer_size:
                idx = random.randint(0, len(buffer) - 1)
                yield buffer.pop(idx)

        # Flush remaining items
        random.shuffle(buffer)
        yield from buffer
```

---

## 7. Samplers

### SequentialSampler

Samples elements sequentially, always in the same order.

```python
from torch.utils.data import SequentialSampler

sampler = SequentialSampler(dataset)
# Produces indices: [0, 1, 2, 3, ..., len(dataset)-1]

loader = DataLoader(dataset, sampler=sampler, batch_size=32)
```

### RandomSampler

Samples elements randomly. If `replacement=False` (default), samples are drawn without replacement. If `replacement=True`, samples can be drawn multiple times.

```python
from torch.utils.data import RandomSampler

# Without replacement (each sample appears exactly once per epoch)
sampler = RandomSampler(dataset)
# Equivalent to: DataLoader(dataset, shuffle=True)

# With replacement
sampler = RandomSampler(
    dataset,
    replacement=True,
    num_samples=1000  # Draw 1000 samples per epoch
)

# With generator for reproducibility
g = torch.Generator()
g.manual_seed(42)
sampler = RandomSampler(dataset, generator=g)

loader = DataLoader(dataset, sampler=sampler, batch_size=32)
```

### SubsetRandomSampler

Samples from a given list of indices randomly (without replacement).

```python
from torch.utils.data import SubsetRandomSampler

# Create train/val split using samplers
dataset_size = len(dataset)
indices = list(range(dataset_size))
np.random.shuffle(indices)

split = int(np.floor(0.2 * dataset_size))
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)
```

### WeightedRandomSampler

Samples elements from `[0, ..., len(weights)-1]` with given probabilities.

```python
from torch.utils.data import WeightedRandomSampler

# Address class imbalance
# Suppose we have 1000 class 0, 100 class 1, 100 class 2
labels = [0] * 1000 + [1] * 100 + [2] * 100
class_counts = [1000, 100, 100]
class_weights = [1.0 / c for c in class_counts]  # [0.001, 0.01, 0.01]
sample_weights = [class_weights[label] for label in labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(labels),
    replacement=True  # Required when oversampling minority classes
)

loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

**WeightedRandomSampler parameters:**
```python
WeightedRandomSampler(
    weights,             # (sequence) sampling weights for each element
    num_samples,         # (int) number of samples drawn per epoch
    replacement=True,    # (bool) draw with replacement
    generator=None,      # (torch.Generator) RNG generator
)
```

### BatchSampler

Wraps another sampler to yield a mini-batch of indices at a time.

```python
from torch.utils.data import BatchSampler, RandomSampler

# Create a batch sampler from a random sampler
sampler = RandomSampler(dataset)
batch_sampler = BatchSampler(sampler, batch_size=32, drop_last=False)

loader = DataLoader(dataset, batch_sampler=batch_sampler)

# Uneven batch sizes using custom BatchSampler
class DynamicBatchSampler(BatchSampler):
    def __init__(self, sampler, max_tokens, max_length):
        self.sampler = sampler
        self.max_tokens = max_tokens
        self.max_length = max_length

    def __iter__(self):
        batch = []
        total_tokens = 0
        for idx in self.sampler:
            length = self._get_length(idx)
            if total_tokens + length > self.max_tokens or len(batch) >= self.max_length:
                if batch:
                    yield batch
                batch = []
                total_tokens = 0
            batch.append(idx)
            total_tokens += length
        if batch:
            yield batch

    def __len__(self):
        raise NotImplementedError
```

### DistributedSampler

Sampler that restricts data loading to a subset of the dataset for distributed training.

```python
from torch.utils.data.distributed import DistributedSampler

# Basic distributed sampler
sampler = DistributedSampler(
    dataset,
    num_replicas=None,    # Total number of processes (default: world_size)
    rank=None,            # Rank of current process (default: rank)
    shuffle=True,         # Shuffle indices across epochs
    seed=0,               # Random seed for shuffling
    drop_last=False,      # Drop tail if not evenly divisible
)

# Usage in distributed training
train_loader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler,
    num_workers=4,
    pin_memory=True,
)

# IMPORTANT: Call set_epoch() at the start of each epoch
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for batch in train_loader:
        train_step(batch)
```

**DistributedSampler parameters:**
```python
DistributedSampler(
    dataset,
    num_replicas=None,         # Number of processes participating
    rank=None,                 # Rank of the current process
    shuffle=True,              # If True, shuffles indices
    seed=0,                    # Random seed used for shuffle
    drop_last=False,           # Drops incomplete last batch
)
```

### DistributedSampler with uneven data

```python
# When dataset size is not divisible by world_size
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    drop_last=False,  # If False, replicates some samples to fill gaps
)
# If len(dataset) % world_size != 0 and drop_last=False,
# the sampler will pad with extra samples to make it evenly divisible
```

---

## 8. Performance Tips

### GPU Utilization Optimization

```python
# Optimal DataLoader configuration for GPU training
loader = DataLoader(
    dataset,
    batch_size=256,            # Largest batch size that fits in GPU memory
    shuffle=True,
    num_workers=8,             # Typically 4 * num_GPUs
    pin_memory=True,           # Faster CPU-to-GPU transfer
    pin_memory_device='cuda:0',# Pin directly to specific device
    persistent_workers=True,   # Avoid worker restart overhead
    prefetch_factor=4,         # Prefetch more batches
    drop_last=True,            # Consistent batch sizes
)
```

### Using non_blocking Transfer

```python
for data, target in loader:
    # non_blocking=True allows overlap of data transfer and computation
    data = data.to('cuda', non_blocking=True)
    target = target.to('cuda', non_blocking=True)

    # ... training step ...
```

### Benchmarking DataLoader Performance

```python
import time

def benchmark_dataloader(loader, num_epochs=3):
    """Measure DataLoader throughput."""
    for epoch in range(num_epochs):
        start = time.time()
        num_samples = 0
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                num_samples += batch[0].size(0)
            elif isinstance(batch, dict):
                first_key = next(iter(batch))
                num_samples += batch[first_key].size(0)
        elapsed = time.time() - start
        throughput = num_samples / elapsed
        print(f"Epoch {epoch}: {elapsed:.2f}s, "
              f"{throughput:.0f} samples/sec, "
              f"{num_samples} samples")

# Benchmark different configurations
configs = [
    {'num_workers': 0, 'pin_memory': False},
    {'num_workers': 4, 'pin_memory': False},
    {'num_workers': 4, 'pin_memory': True},
    {'num_workers': 8, 'pin_memory': True, 'persistent_workers': True},
]

for config in configs:
    loader = DataLoader(dataset, batch_size=64, shuffle=True, **config)
    print(f"\nConfig: {config}")
    benchmark_dataloader(loader)
```

### Avoiding Common Pitfalls

```python
# PITFALL 1: Opening files in __init__ instead of __getitem__
# BAD: All workers share same file handle (can cause issues)
class BadDataset(Dataset):
    def __init__(self, path):
        self.file = open(path, 'rb')  # NOT fork-safe

# GOOD: Open files in __getitem__
class GoodDataset(Dataset):
    def __init__(self, path):
        self.path = path

    def __getitem__(self, idx):
        with open(self.path, 'rb') as f:
            # Seek to the right position or use indexing
            pass

# PITFALL 2: Not using __main__ guard on Windows
# BAD (on Windows/macOS):
# loader = DataLoader(dataset, num_workers=4)

# GOOD:
if __name__ == '__main__':
    loader = DataLoader(dataset, num_workers=4)
    for batch in loader:
        pass

# PITFALL 3: Modifying dataset during iteration
# BAD: Changing dataset between epochs without recreating loader
# GOOD: Recreate sampler/loader if dataset changes

# PITFALL 4: Random state leakage between workers
# BAD: No worker_init_fn
# GOOD: Use worker_init_fn with proper seeding (see Section 5)
```

### Memory-Mapped Files for Large Datasets

```python
import numpy as np

class MMapDataset(Dataset):
    """Use memory-mapped files for datasets larger than RAM."""

    def __init__(self, data_path, labels_path):
        # data.npy is a large file (e.g., 100GB)
        # memory mapping avoids loading everything into RAM
        self.data = np.load(data_path, mmap_mode='r')
        self.labels = np.load(labels_path, mmap_mode='r')

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.data[idx].copy()),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)

# Each worker reads only its needed portion from disk
loader = DataLoader(
    MMapDataset('large_data.npy', 'labels.npy'),
    batch_size=64,
    num_workers=4,
)
```

---

## 9. Advanced Patterns

### DataLoader with Automatic Mixed Precision

```python
from torch.cuda.amp import autocast, GradScaler

loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
scaler = GradScaler()

for data, target in loader:
    data, target = data.to('cuda'), target.to('cuda')
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### DataLoader State Checkpointing

```python
import pickle

def save_dataloader_state(loader, epoch, batch_idx, path):
    """Save the state of a DataLoader for resuming training."""
    state = {
        'epoch': epoch,
        'batch_idx': batch_idx,
    }
    if hasattr(loader, 'sampler') and hasattr(loader.sampler, 'set_epoch'):
        state['sampler_epoch'] = epoch
    with open(path, 'wb') as f:
        pickle.dump(state, f)

def load_dataloader_state(loader, path):
    """Resume DataLoader from a checkpoint."""
    with open(path, 'rb') as f:
        state = pickle.load(f)
    if hasattr(loader, 'sampler') and hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(state['epoch'])
    return state['epoch'], state['batch_idx']
```

### DataLoader with WebDataset (Sharded Files)

```python
# Pattern for loading from sharded tar files
class TarDataset(torch.utils.data.IterableDataset):
    """Load data from a sequence of tar files."""

    def __init__(self, tar_urls):
        super().__init__()
        self.tar_urls = tar_urls

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        urls = self.tar_urls
        if worker_info is not None:
            # Split URLs across workers
            urls = urls[worker_info.id::worker_info.num_workers]
        for url in urls:
            yield from self._read_tar(url)

    def _read_tar(self, url):
        import tarfile
        import io
        # Read and yield samples from tar file
        pass
```

### Combining Multiple DataLoaders

```python
from torch.utils.data import DataLoader

# Zip multiple loaders (e.g., for multi-modal training)
image_loader = DataLoader(image_dataset, batch_size=32, shuffle=True)
text_loader = DataLoader(text_dataset, batch_size=32, shuffle=True)

for (img_batch, img_labels), (txt_batch, txt_labels) in zip(image_loader, text_loader):
    # Train with both modalities
    pass

# Alternate between loaders
from itertools import cycle

def alternating_loader(*loaders):
    """Yield batches from multiple loaders in alternation."""
    iters = [iter(loader) for loader in loaders]
    while True:
        for i, it in enumerate(iters):
            try:
                yield next(it)
            except StopIteration:
                iters[i] = iter(loaders[i])
                yield next(iters[i])
```

### DataLoader with DALI Integration

```python
# Pattern for using NVIDIA DALI with PyTorch
# (Requires nvidia-dali-cudaXXX package)

"""
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from nvidia.dali import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn

@pipeline_def
def create_dali_pipeline(data_dir, batch_size, num_threads, device_id):
    images, labels = fn.readers.file(
        file_root=data_dir,
        random_shuffle=True,
    )
    images = fn.decoders.image(images, device='mixed')
    images = fn.resize(images, resize_x=224, resize_y=224)
    images = fn.normalize(images, mean=[0.485*255, 0.456*255, 0.406*255],
                          stddev=[0.229*255, 0.224*255, 0.225*255])
    return images, labels

pipe = create_dali_pipeline(
    batch_size=32,
    num_threads=4,
    device_id=0,
    data_dir='/path/to/data',
)
pipe.build()
dali_loader = DALIClassificationIterator(pipe, reader_name='Reader')
"""
```

---

## 10. DataLoader Attributes and Methods

### Attributes

```python
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Read-only attributes
loader.dataset          # The dataset object
loader.batch_size       # Batch size (int or None)
loader.num_workers      # Number of worker processes
loader.sampler          # The sampler being used
loader.batch_sampler    # The batch sampler
loader.collate_fn       # The collate function
loader.pin_memory       # Whether pin_memory is enabled
loader.drop_last        # Whether last batch is dropped
loader.timeout          # Timeout for batch collection
loader.persistent_workers  # Whether workers persist
loader.prefetch_factor  # Number of batches prefetched per worker
```

### Iterator Protocol

```python
loader = DataLoader(dataset, batch_size=32)

# Get iterator
it = iter(loader)

# Get next batch
batch = next(it)

# Iterate in a for loop
for batch in loader:
    pass

# DataLoader creates a new iterator each time __iter__ is called
# This re-shuffles (if shuffle=True) and resets workers
```

---

## 11. DataLoader Error Handling

```python
# Common errors and solutions

# Error 1: RuntimeError: DataLoader worker ... exited unexpectedly
# Solution: Check for errors in worker_init_fn or dataset.__getitem__
# Debug with num_workers=0 first

# Error 2: RuntimeError: Cannot re-initialize CUDA in forked subprocess
# Solution: Initialize CUDA after DataLoader, or use multiprocessing_context='spawn'
loader = DataLoader(dataset, num_workers=4, multiprocessing_context='spawn')

# Error 3: OSError: [Errno 24] Too many open files
# Solution: Increase file descriptor limit or reduce num_workers
import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

# Error 4: ValueError: DataLoader with IterableDataset: expected iterable
# Solution: Ensure __iter__ returns an iterator (use yield or return iter())
```

### Graceful Error Handling in Workers

```python
class RobustDataset(Dataset):
    """Dataset that handles errors in __getitem__ gracefully."""

    def __init__(self, data_paths):
        self.data_paths = data_paths
        # Pre-validate data
        self.valid_indices = []
        for i, path in enumerate(data_paths):
            try:
                self._validate(path)
                self.valid_indices.append(i)
            except Exception:
                print(f"Skipping invalid sample: {path}")

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        try:
            return self._load(self.data_paths[actual_idx])
        except Exception as e:
            # Return a dummy sample or raise
            raise RuntimeError(
                f"Error loading sample {actual_idx}: {e}"
            ) from e

    def __len__(self):
        return len(self.valid_indices)
```

---

## 12. Complete Example: End-to-End Training DataLoader

```python
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random

# 1. Define dataset
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=512):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx][:self.max_length]
        token_ids = [self.vocab.get(t, self.vocab['<unk>']) for t in tokens]
        return torch.tensor(token_ids, dtype=torch.long), self.labels[idx]

# 2. Define collate function
def collate_fn(batch):
    token_ids, labels = zip(*batch)
    lengths = torch.tensor([len(ids) for ids in token_ids])
    padded = pad_sequence(token_ids, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    attention_mask = torch.arange(padded.size(1))[None, :] < lengths[:, None]
    return {
        'input_ids': padded,
        'attention_mask': attention_mask.long(),
        'labels': labels,
    }

# 3. Worker initialization
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 4. Create DataLoader
g = torch.Generator()
g.manual_seed(42)

train_loader = DataLoader(
    TextClassificationDataset(train_texts, train_labels, vocab),
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True,
    drop_last=True,
    worker_init_fn=seed_worker,
    generator=g,
    persistent_workers=True,
    prefetch_factor=2,
)

# 5. Training loop
device = torch.device('cuda')
model = MyModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        output = model(input_ids, attention_mask=attention_mask)
        loss = torch.nn.functional.cross_entropy(output, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```
