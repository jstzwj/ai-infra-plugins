# PyTorch - Chapter 56: Backend System

This reference covers the torch.backends module for configuring compute backends.

---

## 56.1 CUDA Backend

```python
torch.backends.cuda.is_built()                    # True if built with CUDA
torch.backends.cuda.matmul.allow_tf32 = True      # Enable TF32 for matmul
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.preferred_blas_library()   # Get preferred BLAS
torch.backends.cuda.preferred_linalg_library()        # Get preferred linalg
torch.backends.cuda.cufft_plan_cache.max_size = 4096   # cuFFT cache size
torch.backends.cuda.cufft_plan_cache.clear()           # Clear cuFFT cache
```

---

## 56.2 cuDNN Backend

```python
torch.backends.cudnn.is_available()
torch.backends.cudnn.enabled = True               # Enable/disable cuDNN
torch.backends.cudnn.benchmark = True              # Auto-tune for best algorithm
torch.backends.cudnn.deterministic = True           # Reproducible results
torch.backends.cudnn.allow_tf32 = True             # Enable TF32 in cuDNN
torch.backends.cudnn.version()                     # cuDNN version
```

---

## 56.3 Other Backends

```python
# MKL (Intel Math Kernel Library)
torch.backends.mkl.is_available()
torch.backends.mkl.enabled
torch.set_num_threads(4)           # Intra-op threads
torch.get_num_threads()
torch.set_num_interop_threads(2)   # Inter-op threads
torch.get_num_interop_threads()

# oneDNN (MKL-DNN)
torch.backends.mkldnn.is_available()
torch.backends.mkldnn.enabled

# OpenMP
torch.backends.openmp.is_available()
torch.backends.openmp.num_threads

# MPS (Apple Silicon)
torch.backends.mps.is_available()
torch.backends.mps.is_built()
```

---

## 56.4 Environment Variables

```bash
OMP_NUM_THREADS=4          # OpenMP threads
MKL_NUM_THREADS=4          # MKL threads
TORCH_NUM_THREADS=4        # PyTorch threads
CUDA_VISIBLE_DEVICES=0,1   # Visible GPUs
TORCH_CUDA_ARCH_LIST="8.0" # Target GPU architectures
```
