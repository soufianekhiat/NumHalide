# Signal Processing

## FFT

`#include "fft.h"`

All FFT functions take/return `Halide::Func` objects that produce a 2-element `Tuple` (real part, imaginary part).

| Function | NumPy | Notes |
|---|---|---|
| `fft(f, N)` | `np.fft.fft(a)` | DFT; N need not be power of 2 |
| `ifft(f, N)` | `np.fft.ifft(a)` | Unnormalized; divide by N for round-trip |
| `ifft_normalized(f, N)` | `np.fft.ifft(a)` | Normalized (÷N); round-trip matches input |
| `fft2d(f, rows, cols)` | `np.fft.fft2(a)` | 2D DFT |
| `ifft2d(f, rows, cols)` | `np.fft.ifft2(a)` | Unnormalized |
| `fftshift_1d(f, N)` | `np.fft.fftshift(a)` | Shift zero-frequency to center |
| `fftshift_2d(f, rows, cols)` | `np.fft.fftshift(a)` | 2D shift |
| `power_spectrum(f, N)` | `np.abs(np.fft.fft(a))**2` | |
| `power_spectrum_2d(f, rows, cols)` | — | 2D variant |

**Inline complex helpers** (for building FFT pipelines):
- `complex(re, im)` — make a Tuple
- `complex_add(a, b)` / `complex_mul(a, b)` / `complex_conj(a)` — Tuple arithmetic
- `complex_mag(a)` — magnitude
- `expj(theta)` — `e^(iθ)` as Tuple

## Fast FFT

`#include "fft_fast.h"`

Optimized Cooley-Tukey with precomputed bit-reversal permutation. Drop-in replacement for `fft`/`ifft` when size is a power of 2.

| Function | NumPy | Notes |
|---|---|---|
| `fft_fast(f, N)` | `np.fft.fft(a)` | N must be power of 2 |
| `ifft_fast(f, N)` | `np.fft.ifft(a)` | N must be power of 2; unnormalized |

## Real FFT

`#include "rfft.h"`

| Function | NumPy | Notes |
|---|---|---|
| `rfft(f, N)` | `np.fft.rfft(a)` | Real input; output size = N/2+1 |
| `irfft(f, N)` | `np.fft.irfft(a)` | N = full signal length |
| `rfft2d(f, rows, cols)` | `np.fft.rfft2(a)` | 2D real FFT |
| `irfft2d(f, rows, cols)` | `np.fft.irfft2(a)` | — |
| `fftfreq(N)` | `np.fft.fftfreq(N)` | Frequency bins for FFT |
| `rfftfreq(N)` | `np.fft.rfftfreq(N)` | Frequency bins for RFFT |

## Spectral Analysis

`#include "fft_ext.h"`

| Function | NumPy / SciPy | Notes |
|---|---|---|
| `cross_power_spectrum(a, b, rows, cols)` | — | Normalized cross-spectrum; phase correlation |
| `spectral_centroid(f, N)` | — | Weighted mean frequency (scalar) |

## Window Functions

`#include "window.h"`

| Function | NumPy | Notes |
|---|---|---|
| `hanning(N)` | `np.hanning(N)` | — |
| `hamming(N)` | `np.hamming(N)` | — |
| `blackman(N)` | `np.blackman(N)` | — |
| `bartlett(N)` | `np.bartlett(N)` | — |
| `kaiser(N, beta)` | `np.kaiser(N, beta)` | — |

All return a 1D `Func`.

## Convolution

`#include "conv.h"`

| Function | NumPy / SciPy | Notes |
|---|---|---|
| `convolve1d(f, shape, kernel, k_size, mode)` | `np.convolve(a, k, mode)` | Modes: `"same"`, `"valid"`, `"full"` |
| `convolve2d(f, shape, kernel, k_rows, k_cols)` | `scipy.signal.convolve2d` | — |
| `convolve2d_separable(f, shape, kx, ky, k_size)` | `scipy.ndimage.convolve` | Faster for separable kernels |
| `correlate2d(f, shape, kernel, k_rows, k_cols)` | `scipy.signal.correlate2d` | Cross-correlation |
| `infer_convolve1d(shape, k_size, mode)` | — | Output shape for 1D convolution |

**Convolution modes:**

| Mode | Output size | Boundary |
|---|---|---|
| `"same"` (default) | `n` | Edge-clamp padding |
| `"valid"` | `n − k + 1` | No padding; kernel fully overlaps |
| `"full"` | `n + k − 1` | Zero-padding on both sides |

**Built-in kernels:**
```cpp
Func box     = box_kernel(size);
Func gauss   = gaussian_kernel_1d(size, sigma);
Func sobel_x = sobel_x_kernel();
Func sobel_y = sobel_y_kernel();
Func lap     = laplacian_kernel();
```
