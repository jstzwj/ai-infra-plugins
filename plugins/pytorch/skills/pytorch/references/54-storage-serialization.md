# PyTorch - Chapter 54: Storage and Serialization

This reference covers tensor storage, torch.save, and torch.load.

---

## 54.1 Storage Types

```python
# Typed Storage (deprecated, use UntypedStorage)
s = torch.FloatStorage(100)       # 100 float32 elements
s = torch.IntStorage(50)          # 50 int32 elements
s = torch.LongStorage(50)         # 50 int64 elements
s = torch.BoolStorage(50)         # 50 bool elements

# UntypedStorage (recommended)
s = torch.UntypedStorage(400)     # 400 bytes

# Operations
s.fill_(0)                         # Fill with value
s.size()                           # Number of elements
s.tolist()                         # Convert to list
s.data_ptr()                       # Raw memory pointer
s.nbytes()                         # Total bytes
s.is_pinned()                      # Check pinned memory
```

---

## 54.2 torch.save

```python
torch.save(obj, f, pickle_module=pickle, pickle_protocol=2,
           _use_new_zipfile_serialization=True)
```

Saves tensors, models, or any Python object.

```python
# Save model state dict
torch.save(model.state_dict(), 'model.pth')

# Save entire model (not recommended)
torch.save(model, 'model.pth')

# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# Save list of tensors
torch.save([t1, t2, t3], 'tensors.pth')
```

---

## 54.3 torch.load

```python
torch.load(f, map_location=None, pickle_module=pickle, *,
           weights_only=False, mmap=False, **pickle_load_args)
```

```python
# Basic load
state = torch.load('model.pth')

# Safe loading (recommended for untrusted files)
state = torch.load('model.pth', weights_only=True)

# Load to specific device
state = torch.load('model.pth', map_location='cuda:0')
state = torch.load('model.pth', map_location={'cuda:0': 'cuda:1'})
state = torch.load('model.pth', map_location=lambda storage, loc: storage)

# Memory-mapped loading (for large files)
state = torch.load('large_model.pth', mmap=True)
```

---

## 54.4 weights_only=True

When True, only loads tensors, primitive types, and safe types. Prevents arbitrary code execution.

```python
# Safe loading (PyTorch 2.0+)
torch.load('model.pth', weights_only=True)
```
