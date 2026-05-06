# PyTorch - Chapter 59: Device Management

This reference covers device creation, multi-device management, and device-agnostic patterns.

---

## 59.1 torch.device

```python
device = torch.device('cuda:0')
device = torch.device('cpu')
device.type     # 'cuda', 'cpu', 'xpu', 'mps'
device.index    # 0, 1, ... or None

torch.set_default_device('cuda')
torch.get_default_device()
```

---

## 59.2 CUDA Device Management

```python
torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.current_device()
torch.cuda.set_device(0)
torch.cuda.synchronize()

# Device properties
props = torch.cuda.get_device_properties(0)
props.name, props.total_memory, props.major, props.minor

# Memory info
torch.cuda.memory_allocated()
torch.cuda.max_memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.empty_cache()
```

---

## 59.3 Other Backends

```python
# Intel XPU
torch.xpu.is_available()
torch.xpu.device_count()

# Apple MPS
torch.backends.mps.is_available()

# Device-agnostic pattern
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```
