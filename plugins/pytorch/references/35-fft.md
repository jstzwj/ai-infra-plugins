# torch.fft - Fast Fourier Transform

PyTorch FFT module for frequency-domain signal processing. All functions support batching and autograd.

```python
import torch
```

---

## 1D Transforms

### torch.fft.fft / ifft

```python
torch.fft.fft(
    input: Tensor,                   # [..., N] complex or real
    n: int = None,                   # output length (zero-pad/truncate)
    dim: int = -1,                   # transform dimension
    norm: str = None,                # "forward" | "backward" | "ortho"
) -> Tensor                          # complex tensor

torch.fft.ifft(input, n=None, dim=-1, norm=None) -> Tensor
```

```python
x = torch.randn(8)
X = torch.fft.fft(x)           # [8] complex
x_back = torch.fft.ifft(X)     # [8] complex, approx equals x
assert torch.allclose(x, x_back.real, atol=1e-6)
```

### torch.fft.rfft / irfft

Real-valued input/output (exploits Hermitian symmetry).

```python
torch.fft.rfft(input: Tensor, n=None, dim=-1, norm=None) -> Tensor
# input: real [..., N], output: complex [..., N//2+1]

torch.fft.irfft(input: Tensor, n=None, dim=-1, norm=None) -> Tensor
# input: complex [..., N//2+1], output: real [..., N]
```

```python
x = torch.randn(8)
X = torch.fft.rfft(x)          # complex [5] (8//2+1)
x_back = torch.fft.irfft(X, n=8)  # real [8]
assert torch.allclose(x, x_back, atol=1e-6)
```

### torch.fft.hfft / ihfft

```python
torch.fft.hfft(input, n=None, dim=-1, norm=None) -> Tensor      # complex -> real
torch.fft.ihfft(input, n=None, dim=-1, norm=None) -> Tensor     # real -> complex
```

---

## Frequency Bins

```python
torch.fft.fftfreq(n: int, d: float = 1.0, *, dtype=None, device=None) -> Tensor
# Frequency bins for FFT of length n. d = sample spacing.

torch.fft.rfftfreq(n: int, d: float = 1.0, *, dtype=None, device=None) -> Tensor
# Frequency bins for RFFT of length n (non-negative only).

torch.fft.fftshift(input: Tensor, dim=None) -> Tensor    # shift zero to center
torch.fft.ifftshift(input: Tensor, dim=None) -> Tensor   # inverse shift
```

```python
freqs = torch.fft.fftfreq(8, d=0.01)    # [-0.5, -0.375, ..., 0.375]
rfreqs = torch.fft.rfftfreq(8, d=0.01)  # [0, 0.125, 0.25, 0.375, 0.5]

X = torch.fft.fftshift(torch.fft.fft(x))  # center zero frequency for visualization
```

---

## 2D Transforms

```python
torch.fft.fft2(input, s=None, dim=(-2,-1), norm=None) -> Tensor
torch.fft.ifft2(input, s=None, dim=(-2,-1), norm=None) -> Tensor
torch.fft.rfft2(input, s=None, dim=(-2,-1), norm=None) -> Tensor
torch.fft.irfft2(input, s=None, dim=(-2,-1), norm=None) -> Tensor
torch.fft.hfft2(input, s=None, dim=(-2,-1), norm=None) -> Tensor
torch.fft.ihfft2(input, s=None, dim=(-2,-1), norm=None) -> Tensor
```

```python
img = torch.randn(32, 32)
F = torch.fft.rfft2(img)             # complex [32, 17]
img_back = torch.fft.irfft2(F, s=(32, 32))  # real [32, 32]
assert torch.allclose(img, img_back, atol=1e-5)

# Shifted 2D spectrum
F_shifted = torch.fft.fftshift(F)
```

---

## N-D Transforms

```python
torch.fft.fftn(input, s=None, dim=None, norm=None) -> Tensor
torch.fft.ifftn(input, s=None, dim=None, norm=None) -> Tensor
torch.fft.rfftn(input, s=None, dim=None, norm=None) -> Tensor
torch.fft.irfftn(input, s=None, dim=None, norm=None) -> Tensor
torch.fft.hfftn(input, s=None, dim=None, norm=None) -> Tensor
torch.fft.ihfftn(input, s=None, dim=None, norm=None) -> Tensor
```

```python
vol = torch.randn(16, 16, 16)
F = torch.fft.rfftn(vol)              # complex [16, 16, 9]
vol_back = torch.fft.irfftn(F, s=(16, 16, 16))
```

---

## Short-Time Fourier Transform

### torch.fft.stft

```python
torch.fft.stft(
    input: Tensor,                   # [..., N] real or complex
    n_fft: int,                      # FFT size
    hop_length: int = None,          # default: n_fft // 4
    win_length: int = None,          # default: n_fft
    window: Tensor = None,           # window function
    center: bool = True,             # pad input so frames centered
    pad_mode: str = "reflect",       # padding mode
    normalized: bool = False,
    onesided: bool = True,           # only positive frequencies
    return_complex: bool = None,     # must be True for complex output
) -> Tensor                          # [..., n_fft//2+1, T] if onesided
```

```python
signal = torch.randn(16000)         # 1 second at 16kHz
spec = torch.fft.stft(signal, n_fft=512, hop_length=128,
                       window=torch.hann_window(512),
                       return_complex=True)
print(spec.shape)  # [257, 121] (freq_bins x time_frames)

mag = spec.abs()   # magnitude spectrogram
phase = spec.angle()  # phase spectrogram
```

### torch.fft.istft

```python
torch.fft.istft(
    input: Tensor, n_fft: int, hop_length=None, win_length=None,
    window=None, center=True, normalized=False, onesided=True,
    length: int = None, return_complex: bool = False,
) -> Tensor
```

```python
recon = torch.fft.istft(spec, n_fft=512, hop_length=128,
                         window=torch.hann_window(512), length=16000)
assert torch.allclose(signal, recon, atol=1e-4)
```

---

## Window Functions

```python
from torch.signal.windows import (
    hann_window, hamming_window, blackman_window,
    kaiser_window, cosine_window, exponential_window,
    gaussian_window, general_cosine, general_hamming,
)

hann = torch.hann_window(512)
hamming = torch.hamming_window(512)
blackman = torch.blackman_window(512)
```

---

## Normalization Modes

| norm | Forward | Inverse |
|------|---------|---------|
| `"backward"` (default) | No normalization | 1/n |
| `"forward"` | 1/n | No normalization |
| `"ortho"` | 1/sqrt(n) | 1/sqrt(n) |

```python
X = torch.fft.fft(x, norm="ortho")       # unitary transform
x_back = torch.fft.ifft(X, norm="ortho")  # perfectly unitary
```

---

## Common Patterns

### Spectral Filtering (Low-Pass)

```python
x = torch.randn(1024)
X = torch.fft.rfft(x)
freqs = torch.fft.rfftfreq(1024)
mask = (freqs < 0.1).to(X.dtype)   # keep frequencies below 0.1
X_filtered = X * mask
x_filtered = torch.fft.irfft(X_filtered, n=1024)
```

### Convolution via FFT

```python
def fft_conv(signal, kernel):
    n = signal.size(-1) + kernel.size(-1) - 1
    S = torch.fft.rfft(signal, n=n)
    K = torch.fft.rfft(kernel, n=n)
    return torch.fft.irfft(S * K, n=n)

sig = torch.randn(1024)
ker = torch.randn(64)
result = fft_conv(sig, ker)
```

### Batched FFT

```python
batch = torch.randn(16, 3, 256)     # [B, C, N]
F = torch.fft.rfft(batch, dim=-1)   # FFT along last dim
print(F.shape)                       # [16, 3, 129]
```

### Power Spectrum

```python
x = torch.randn(2048)
X = torch.fft.rfft(x)
power = X.abs() ** 2
freqs = torch.fft.rfftfreq(2048)

# Plot power spectrum
# plt.plot(freqs, power.log())
```

### Spectrogram

```python
audio = torch.randn(2, 16000)         # 2 channels, 1 sec at 16kHz
window = torch.hann_window(400)
spec = torch.fft.stft(audio, n_fft=400, hop_length=160,
                       window=window, return_complex=True)
# spec shape: [2, 201, 101]  (channels x freq_bins x time_frames)
magnitude = spec.abs()
log_spec = magnitude.log()
```
