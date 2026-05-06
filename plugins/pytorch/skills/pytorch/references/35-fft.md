# PyTorch FFT Operations - Comprehensive Reference

This chapter covers all Fourier transform operations in `torch.fft`, including 1D, 2D, and N-dimensional FFTs, real-valued transforms, Short-Time Fourier Transform (STFT), window functions, frequency utilities, and normalization modes.

---

## 1. One-Dimensional FFT

### torch.fft.fft

Computes the one-dimensional discrete Fourier Transform.

```python
torch.fft.fft(
    input,              # (Tensor) the input tensor
    n=None,             # (int) signal length (pads/truncates if needed)
    dim=-1,             # (int) dimension along which to compute FFT
    norm=None,          # (str) normalization mode: "forward", "backward", "ortho"
    *, out=None,
)
```

```python
import torch
import torch.fft as fft

# Basic 1D FFT
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
X = fft.fft(x)
print(X)
# tensor([10.+0.j, -2.+2.j, -2.+0.j, -2.-2.j])

# FFT with specific output length (zero-padding)
X = fft.fft(x, n=8)  # Pad to 8 points
print(X.shape)  # torch.Size([8])

# FFT along a specific dimension
x = torch.randn(3, 64)  # 3 signals of length 64
X = fft.fft(x, dim=-1)  # FFT along last dimension
print(X.shape)  # torch.Size([3, 64])

# Batched FFT
x = torch.randn(16, 128)
X = fft.fft(x, dim=-1)
print(X.shape)  # torch.Size([16, 128])
```

### torch.fft.ifft

Computes the one-dimensional inverse discrete Fourier Transform.

```python
torch.fft.ifft(
    input,              # (Tensor) the FFT input tensor
    n=None,             # (int) signal length
    dim=-1,             # (int) dimension for IFFT
    norm=None,          # (str) normalization mode
    *, out=None,
)
```

```python
# Round-trip: FFT then IFFT
x = torch.randn(64)
X = fft.fft(x)
x_reconstructed = fft.ifft(X)
print(torch.allclose(x, x_reconstructed.real))  # True

# IFFT with different output length
X = torch.randn(32, dtype=torch.complex64)
x = fft.ifft(X, n=64)  # Output length 64
print(x.shape)  # torch.Size([64])
```

---

## 2. Real-Valued FFT

### torch.fft.rfft

Computes the one-dimensional FFT of a real-valued input, returning only the non-redundant half.

```python
torch.fft.rfft(
    input,              # (Tensor) real-valued input tensor
    n=None,             # (int) signal length
    dim=-1,             # (int) dimension along which to compute
    norm=None,          # (str) normalization mode
    *, out=None,
)
```

```python
# Real FFT returns only half the spectrum (Hermitian symmetry)
x = torch.randn(64)
X = fft.rfft(x)
print(X.shape)  # torch.Size([33])  (N//2 + 1 = 33)
print(X.dtype)  # torch.complex64

# rfft vs fft
X_full = fft.fft(x)
X_half = fft.rfft(x)
print(torch.allclose(X_full[:33], X_half))  # True

# With output length
x = torch.randn(32)
X = fft.rfft(x, n=64)  # Zero-pad to 64, output length 33
print(X.shape)  # torch.Size([33])
```

### torch.fft.irfft

Computes the inverse of `rfft`. Takes half the spectrum and returns a real-valued signal.

```python
torch.fft.irfft(
    input,              # (Tensor) the FFT input (complex)
    n=None,             # (int) output signal length
    dim=-1,             # (int) dimension
    norm=None,          # (str) normalization mode
    *, out=None,
)
```

```python
# Round-trip: rfft then irfft
x = torch.randn(64)
X = fft.rfft(x)
x_reconstructed = fft.irfft(X, n=64)
print(torch.allclose(x, x_reconstructed))  # True

# Important: specify n for odd-length signals
X = torch.randn(33, dtype=torch.complex64)
x = fft.irfft(X, n=64)  # Must specify n when signal length is even
print(x.shape)  # torch.Size([64])
print(x.dtype)  # torch.float32
```

---

## 3. Hermitian FFT

### torch.fft.hfft

Computes the FFT of a Hermitian-symmetric (conjugate-symmetric) input, returning a real-valued output.

```python
torch.fft.hfft(
    input,              # (Tensor) Hermitian-symmetric input
    n=None,             # (int) output signal length
    dim=-1,             # (int) dimension
    norm=None,          # (str) normalization mode
    *, out=None,
)
```

```python
# HFFT: Hermitian input -> real output
x = torch.randn(64)
X = fft.rfft(x)   # Real input -> half spectrum (Hermitian)
x_back = fft.hfft(X, n=64)  # Hermitian input -> real output
print(torch.allclose(x, x_back))  # True
```

### torch.fft.ihfft

Computes the inverse of `hfft`. Takes a real-valued signal and returns a Hermitian-symmetric half-spectrum.

```python
torch.fft.ihfft(
    input,              # (Tensor) real-valued input
    n=None,             # (int) signal length
    dim=-1,             # (int) dimension
    norm=None,          # (str) normalization mode
    *, out=None,
)
```

```python
# ihfft: real input -> Hermitian half-spectrum
x = torch.randn(64)
X = fft.ihfft(x)
print(X.shape)  # torch.Size([33])
print(X.dtype)  # torch.complex64

# Round-trip
x_back = fft.hfft(X, n=64)
print(torch.allclose(x, x_back))  # True
```

---

## 4. Frequency Utilities

### torch.fft.fftfreq

Returns the discrete Fourier Transform sample frequencies.

```python
torch.fft.fftfreq(
    n,              # (int) window length
    d=1.0,          # (float) sample spacing
    *, out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
)
```

```python
# Frequencies for a 8-point FFT
freqs = fft.fftfreq(8)
print(freqs)
# tensor([ 0.0000,  0.1250,  0.2500,  0.3750, -0.5000, -0.3750, -0.2500, -0.1250])

# With custom sample spacing (e.g., sampling rate = 1000 Hz)
freqs = fft.fftfreq(8, d=1.0/1000)
print(freqs)
# tensor([   0.,  125.,  250.,  375., -500., -375., -250., -125.])
```

### torch.fft.rfftfreq

Returns the discrete Fourier Transform sample frequencies for `rfft`.

```python
torch.fft.rfftfreq(
    n,              # (int) window length
    d=1.0,          # (float) sample spacing
    *, out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
)
```

```python
# Frequencies for rfft output (only positive frequencies)
freqs = fft.rfftfreq(8)
print(freqs)
# tensor([0.0000, 0.1250, 0.2500, 0.3750, 0.5000])

# For a signal sampled at 44100 Hz
freqs = fft.rfftfreq(1024, d=1.0/44100)
print(freqs[:5])  # [0., 43.0664, 86.1328, 129.1992, 172.2656]
```

### torch.fft.fftshift

Shifts the zero-frequency component to the center of the spectrum.

```python
torch.fft.fftshift(
    input,          # (Tensor) input tensor
    dim=None,       # (int or tuple) dimensions to shift
)
```

```python
# Shift zero frequency to center
x = torch.arange(8)
X = fft.fft(x)
X_shifted = fft.fftshift(X)

# Compare frequency arrays
freqs = fft.fftfreq(8)
freqs_shifted = fft.fftshift(freqs)
print(freqs_shifted)
# tensor([-0.5000, -0.3750, -0.2500, -0.1250,  0.0000,  0.1250,  0.2500,  0.3750])
```

### torch.fft.ifftshift

The inverse of `fftshift`.

```python
torch.fft.ifftshift(
    input,          # (Tensor) input tensor
    dim=None,       # (int or tuple) dimensions to shift
)
```

```python
# Round-trip
freqs = fft.fftfreq(8)
freqs_shifted = fft.fftshift(freqs)
freqs_back = fft.ifftshift(freqs_shifted)
print(torch.allclose(freqs, freqs_back))  # True
```

---

## 5. Two-Dimensional FFT

### torch.fft.fft2

Computes the two-dimensional discrete Fourier Transform.

```python
torch.fft.fft2(
    input,              # (Tensor) input tensor (at least 2D)
    s=None,             # (tuple) signal size in the transform dimensions
    dim=(-2, -1),       # (tuple) dimensions along which to compute
    norm=None,          # (str) normalization mode
    *, out=None,
)
```

```python
# 2D FFT on an image-like tensor
image = torch.randn(3, 64, 64)  # 3-channel 64x64 image

# FFT on spatial dimensions
F = fft.fft2(image, dim=(-2, -1))
print(F.shape)  # torch.Size([3, 64, 64])
print(F.dtype)  # torch.complex64

# With custom output size
F = fft.fft2(image, s=(128, 128))  # Zero-pad to 128x128
print(F.shape)  # torch.Size([3, 128, 128])

# Shift zero frequency to center
F_shifted = fft.fftshift(F, dim=(-2, -1))
```

### torch.fft.ifft2

Computes the 2D inverse FFT.

```python
torch.fft.ifft2(
    input,              # (Tensor) the FFT input
    s=None,             # (tuple) signal size
    dim=(-2, -1),       # (tuple) dimensions
    norm=None,          # (str) normalization mode
    *, out=None,
)
```

```python
# Round-trip
image = torch.randn(64, 64)
F = fft.fft2(image)
image_back = fft.ifft2(F)
print(torch.allclose(image, image_back.real))  # True
```

### torch.fft.rfft2

2D FFT for real-valued input.

```python
torch.fft.rfft2(
    input,              # (Tensor) real-valued input
    s=None,             # (tuple) signal size
    dim=(-2, -1),       # (tuple) dimensions
    norm=None,          # (str) normalization mode
    *, out=None,
)
```

```python
image = torch.randn(64, 64)
F = fft.rfft2(image)
print(F.shape)  # torch.Size([64, 33])  (Hermitian in last dim)

# With padding
F = fft.rfft2(image, s=(128, 128))
print(F.shape)  # torch.Size([128, 65])
```

### torch.fft.irfft2

Inverse of `rfft2`.

```python
torch.fft.irfft2(
    input,              # (Tensor) complex FFT input
    s=None,             # (tuple) output signal size
    dim=(-2, -1),       # (tuple) dimensions
    norm=None,          # (str) normalization mode
    *, out=None,
)
```

```python
image = torch.randn(64, 64)
F = fft.rfft2(image)
image_back = fft.irfft2(F, s=(64, 64))
print(torch.allclose(image, image_back))  # True
```

### torch.fft.hfft2 and torch.fft.ihfft2

```python
# hfft2: Hermitian input -> real output (2D)
# ihfft2: real input -> Hermitian output (2D)
image = torch.randn(64, 64)
X = fft.ihfft2(image)      # torch.Size([64, 33])
image_back = fft.hfft2(X, s=(64, 64))  # torch.Size([64, 64])
```

---

## 6. N-Dimensional FFT

### torch.fft.fftn

Computes the N-dimensional discrete Fourier Transform.

```python
torch.fft.fftn(
    input,              # (Tensor) input tensor
    s=None,             # (tuple) signal size in each transform dimension
    dim=None,           # (tuple) dimensions along which to compute
    norm=None,          # (str) normalization mode
    *, out=None,
)
```

```python
# 3D FFT
volume = torch.randn(32, 32, 32)
F = fft.fftn(volume)
print(F.shape)  # torch.Size([32, 32, 32])

# FFT along specific dimensions only
F = fft.fftn(volume, dim=(0, 1))  # FFT only along first two dims
print(F.shape)  # torch.Size([32, 32, 32])

# With custom output sizes
F = fft.fftn(volume, s=(64, 64, 64))  # Zero-pad all dims
print(F.shape)  # torch.Size([64, 64, 64])
```

### torch.fft.ifftn

Computes the N-dimensional inverse FFT.

```python
torch.fft.ifftn(
    input,              # (Tensor) FFT input
    s=None,             # (tuple) signal size
    dim=None,           # (tuple) dimensions
    norm=None,          # (str) normalization mode
    *, out=None,
)
```

```python
# Round-trip
volume = torch.randn(32, 32, 32)
F = fft.fftn(volume)
volume_back = fft.ifftn(F)
print(torch.allclose(volume, volume_back.real))  # True
```

### torch.fft.rfftn

N-dimensional FFT for real-valued input.

```python
torch.fft.rfftn(
    input,              # (Tensor) real-valued input
    s=None,             # (tuple) signal size
    dim=None,           # (tuple) dimensions
    norm=None,          # (str) normalization mode
    *, out=None,
)
```

```python
volume = torch.randn(32, 32, 32)
F = fft.rfftn(volume)
print(F.shape)  # torch.Size([32, 32, 17])  (Hermitian in last dim)
```

### torch.fft.irfftn

Inverse of `rfftn`.

```python
torch.fft.irfftn(
    input,              # (Tensor) complex FFT input
    s=None,             # (tuple) output signal size
    dim=None,           # (tuple) dimensions
    norm=None,          # (str) normalization mode
    *, out=None,
)
```

```python
volume = torch.randn(32, 32, 32)
F = fft.rfftn(volume)
volume_back = fft.irfftn(F, s=(32, 32, 32))
print(torch.allclose(volume, volume_back))  # True
```

### torch.fft.hfftn and torch.fft.ihfftn

```python
# N-dimensional Hermitian transforms
volume = torch.randn(32, 32, 32)
X = fft.ihfftn(volume)
print(X.shape)  # torch.Size([32, 32, 17])
volume_back = fft.hfftn(X, s=(32, 32, 32))
```

---

## 7. Short-Time Fourier Transform (STFT)

### torch.fft.stft

Computes the Short-Time Fourier Transform.

```python
torch.fft.stft(
    input,              # (Tensor) input tensor
    n_fft,              # (int) size of Fourier transform
    hop_length=None,    # (int) distance between sliding window frames
    win_length=None,    # (int) window size (defaults to n_fft)
    window=None,        # (Tensor) window tensor
    center=True,        # (bool) whether to pad input on both sides
    pad_mode='reflect', # (str) padding mode
    normalized=False,   # (bool) whether to return normalized STFT
    onesided=None,      # (bool) whether to return only positive frequencies
    return_complex=True,# (bool) whether to return complex tensor
)
```

```python
# Basic STFT
signal = torch.randn(16000)  # 1 second at 16kHz
n_fft = 400
hop_length = 160
window = torch.hann_window(n_fft)

S = fft.stft(signal, n_fft=n_fft, hop_length=hop_length, window=window)
print(S.shape)  # torch.Size([201, 101])  (n_fft//2+1, num_frames)

# STFT on batched audio
batch = torch.randn(4, 16000)  # 4 audio clips
S = fft.stft(batch, n_fft=400, hop_length=160, window=torch.hann_window(400))
print(S.shape)  # torch.Size([4, 201, 101])

# Parameters explained:
# n_fft=400: FFT size (also determines frequency resolution)
# hop_length=160: 10ms hop at 16kHz
# win_length defaults to n_fft
# center=True: pads signal so frame t is centered at t * hop_length
```

### torch.fft.istft

Computes the inverse Short-Time Fourier Transform.

```python
torch.fft.istft(
    input,              # (Tensor) STFT tensor (complex)
    n_fft,              # (int) size of Fourier transform
    hop_length=None,    # (int) hop length
    win_length=None,    # (int) window size
    window=None,        # (Tensor) window tensor
    center=True,        # (bool) whether STFT was centered
    normalized=False,   # (bool) whether STFT was normalized
    onesided=None,      # (bool) whether STFT was onesided
    length=None,        # (int) desired output length
    return_complex=False,
)
```

```python
# Round-trip: STFT then ISTFT
signal = torch.randn(16000)
n_fft = 400
hop_length = 160
window = torch.hann_window(n_fft)

S = fft.stft(signal, n_fft=n_fft, hop_length=hop_length, window=window)
signal_back = fft.istft(S, n_fft=n_fft, hop_length=hop_length, window=window, length=16000)
print(torch.allclose(signal, signal_back, atol=1e-5))  # True (approximately)

# ISTFT with length specification
signal_back = fft.istft(S, n_fft=n_fft, hop_length=hop_length, window=window, length=8000)
print(signal_back.shape)  # torch.Size([8000])
```

---

## 8. Window Functions

### torch.fft.hann_window

Hann window function.

```python
torch.fft.hann_window(
    window_length,          # (int) size of the window
    periodic=True,          # (bool) if True, creates periodic window for FFT
    *, dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
)
```

```python
# Periodic Hann window (for FFT use)
window = fft.hann_window(400, periodic=True)
print(window.shape)  # torch.Size([400])
print(window.sum())  # Approximately window_length / 2

# Symmetric Hann window (for filter design)
window = fft.hann_window(400, periodic=False)
```

### torch.fft.hamming_window

Hamming window function.

```python
torch.fft.hamming_window(
    window_length,
    periodic=True,
    alpha=0.54,         # coefficient
    beta=0.46,          # coefficient
    *, dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
)
```

```python
window = fft.hamming_window(400, periodic=True)
print(window.shape)  # torch.Size([400])
```

### torch.fft.blackman_window

Blackman window function.

```python
torch.fft.blackman_window(
    window_length,
    periodic=True,
    *, dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
)
```

```python
# Blackman window (better side-lobe suppression than Hann)
window = fft.blackman_window(400, periodic=True)
```

### torch.fft.kaiser_window

Kaiser window function.

```python
torch.fft.kaiser_window(
    window_length,
    periodic=True,
    beta=12.0,          # shape parameter (higher = narrower main lobe)
    *, dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
)
```

```python
# Kaiser window with adjustable beta parameter
window = fft.kaiser_window(400, periodic=True, beta=14.0)
# beta=0: rectangular window
# beta=5: similar to Hamming
# beta=6: similar to Hann
# beta=8.6: similar to Blackman
# beta=14: very narrow main lobe
```

---

## 9. Normalization Modes

All FFT functions accept a `norm` parameter controlling normalization:

| `norm` | Forward (FFT) | Inverse (IFFT) |
|--------|--------------|----------------|
| `"backward"` (default) | No normalization | Normalized by `1/n` |
| `"forward"` | Normalized by `1/n` | No normalization |
| `"ortho"` | Normalized by `1/sqrt(n)` | Normalized by `1/sqrt(n)` |

```python
x = torch.randn(64)

# Default ("backward"): FFT unnormalized, IFFT normalized by 1/n
X = fft.fft(x, norm="backward")
x_back = fft.ifft(X, norm="backward")

# "forward": FFT normalized by 1/n, IFFT unnormalized
X = fft.fft(x, norm="forward")
x_back = fft.ifft(X, norm="forward")

# "ortho": both normalized by 1/sqrt(n) (unitary transform)
X = fft.fft(x, norm="ortho")
x_back = fft.ifft(X, norm="ortho")

# All norms give exact round-trip
assert torch.allclose(x, x_back.real)
```

---

## 10. Practical Examples

### Spectrum Analysis

```python
import torch
import torch.fft as fft

# Generate a signal with known frequencies
sr = 1000  # Sample rate
t = torch.linspace(0, 1, sr, endpoint=False)
freq1, freq2 = 50, 120  # Hz

signal = 1.0 * torch.sin(2 * 3.14159 * freq1 * t) + \
         0.5 * torch.sin(2 * 3.14159 * freq2 * t)

# Compute FFT
X = fft.rfft(signal)
freqs = fft.rfftfreq(sr, d=1.0/sr)

# Compute magnitude spectrum
magnitude = X.abs()

# Find dominant frequencies
top_k = torch.topk(magnitude, 5)
top_freqs = freqs[top_k.indices]
print(f"Dominant frequencies: {top_freqs}")
```

### Filtering in Frequency Domain

```python
def lowpass_filter(signal, cutoff_freq, sample_rate):
    """Apply a low-pass filter in the frequency domain."""
    n = signal.shape[-1]
    X = fft.rfft(signal)
    freqs = fft.rfftfreq(n, d=1.0/sample_rate)

    # Create frequency mask
    mask = freqs <= cutoff_freq
    X_filtered = X * mask.to(X.dtype)

    return fft.irfft(X_filtered, n=n)

# Usage
signal = torch.randn(4096)
filtered = lowpass_filter(signal, cutoff_freq=100, sample_rate=1000)
print(filtered.shape)  # torch.Size([4096])
```

### Convolution via FFT

```python
def fft_convolve(signal, kernel):
    """Compute linear convolution using FFT."""
    n_signal = signal.shape[-1]
    n_kernel = kernel.shape[-1]
    n_fft = n_signal + n_kernel - 1

    # Pad to next power of 2 for efficiency
    n_fft = int(2 ** torch.ceil(torch.log2(torch.tensor(float(n_fft)))))

    # Compute in frequency domain
    S = fft.rfft(signal, n=n_fft)
    K = fft.rfft(kernel, n=n_fft)
    Y = S * K

    # Convert back
    result = fft.irfft(Y, n=n_fft)

    # Return only the valid part (linear convolution)
    return result[:n_signal + n_kernel - 1]

# Usage
signal = torch.randn(1024)
kernel = torch.randn(31)  # FIR filter
output = fft_convolve(signal, kernel)
```

### Spectrogram Computation

```python
def compute_spectrogram(waveform, n_fft=400, hop_length=160, power=2.0):
    """Compute power spectrogram from waveform."""
    window = torch.hann_window(n_fft, device=waveform.device)

    # STFT
    S = fft.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )

    # Power spectrogram
    if power is not None:
        spectrogram = S.abs().pow(power)
    else:
        spectrogram = S

    return spectrogram

# Usage
waveform = torch.randn(16000)  # 1 second at 16kHz
spec = compute_spectrogram(waveform, n_fft=400, hop_length=160)
print(spec.shape)  # torch.Size([201, 97])
```

### Phase Correlation (Image Registration)

```python
def phase_correlate(image1, image2):
    """Estimate translation between two images using phase correlation."""
    # Compute 2D FFT
    F1 = fft.fft2(image1)
    F2 = fft.fft2(image2)

    # Cross-power spectrum
    cross_power = F1 * F2.conj()
    cross_power = cross_power / (cross_power.abs() + 1e-8)

    # Inverse FFT to get correlation
    correlation = fft.ifft2(cross_power).real

    # Find peak (translation offset)
    flat_idx = correlation.argmax()
    peak_y = flat_idx // correlation.shape[-1]
    peak_x = flat_idx % correlation.shape[-1]

    # Handle wrap-around
    h, w = correlation.shape[-2], correlation.shape[-1]
    if peak_y > h // 2:
        peak_y -= h
    if peak_x > w // 2:
        peak_x -= w

    return peak_y.item(), peak_x.item()
```

### Audio Denoising via Spectral Gating

```python
def spectral_gate(waveform, n_fft=1024, hop_length=256, threshold_db=-40):
    """Simple spectral gating noise reduction."""
    window = torch.hann_window(n_fft, device=waveform.device)

    # STFT
    S = fft.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window)

    # Compute magnitude
    magnitude = S.abs()

    # Compute threshold
    threshold = magnitude.mean() * 10 ** (threshold_db / 20)

    # Create mask: keep components above threshold
    mask = (magnitude > threshold).float()

    # Apply mask
    S_clean = S * mask

    # ISTFT
    clean = fft.istft(
        S_clean, n_fft=n_fft, hop_length=hop_length,
        window=window, length=waveform.shape[-1]
    )

    return clean
```

### Cross-Correlation

```python
def fft_cross_correlation(signal1, signal2):
    """Compute normalized cross-correlation using FFT."""
    n1, n2 = signal1.shape[-1], signal2.shape[-1]
    n = n1 + n2 - 1

    # Compute FFT
    F1 = fft.rfft(signal1, n=n)
    F2 = fft.rfft(signal2, n=n)

    # Cross-correlation = IFFT(F1 * conj(F2))
    corr = fft.irfft(F1 * F2.conj(), n=n)

    # Normalize
    norm_factor = torch.sqrt(
        signal1.pow(2).sum() * signal2.pow(2).sum()
    )
    corr = corr / (norm_factor + 1e-8)

    return corr
```

---

## 11. CUDA Considerations

```python
# FFT operations work on CUDA tensors
if torch.cuda.is_available():
    x = torch.randn(4096, device='cuda')
    X = fft.fft(x)  # Computed on GPU
    x_back = fft.ifft(X)

# cuFFT is used under the hood on CUDA
# Performance tips:
# - FFT sizes that are powers of 2 are most efficient
# - Sizes that factor into small primes (2, 3, 5, 7) are also efficient
# - Plan caching is automatic in PyTorch

# For repeated FFTs of the same size, cuFFT plans are cached automatically
# No manual plan management needed
```
