# NumHalide

[NumHalide](https://github.com/soufianekhiat/NumHalide) is a header-only C++20 library providing NumPy-like API built on [Halide](https://halide-lang.org/). It enables expressive array programming with automatic optimization through Halide's compiler.

## Features

### Core Array Operations
- **Factory Functions**: `zeros`, `ones`, `full`, `linspace`, `arange`, `eye`, `meshgrid`
- **Random Generation**: `rand_uniform`, `rand_normal`, `rand_int` with seed control
- **Extended Random**: `rand_exponential`, `rand_bernoulli`, `rand_choice`
- **Shape Manipulation**: `reshape`, `transpose`, `expand_dims`, `squeeze`, `moveaxis`, `flip`, `flipud`, `fliplr`, `rot90`, `roll`, `tile`, `repeat`, `pad`
- **Slicing**: `slice`, `take` with NumPy-style indexing
- **Splitting**: `split`, `split_at`, `hsplit`, `vsplit`
- **Stacking**: `concat`, `stack`, `vstack`, `hstack`
- **Broadcasting**: Automatic shape broadcasting for binary operations

### Reductions and Statistics
- **Reductions**: `sum`, `mean`, `min`, `max`, `prod` with axis support
- **Statistics**: `var`, `std` with axis support and ddof parameter
- **Extended Statistics**: `median`, `ptp`, `average`, `histogram`, `digitize`
- **Boolean Reductions**: `reduce_any`, `reduce_all`, `count_nonzero`
- **Cumulative**: `cumsum`, `cumprod`, `diff`

### Comparisons and Logic
- **Comparisons**: `equal`, `not_equal`, `greater`, `less`, `greater_equal`, `less_equal`
- **Logical Operations**: `logical_and`, `logical_or`, `logical_not`, `logical_xor`
- **Special Value Detection**: `isnan_func`, `isinf_func`, `isfinite_func`, `isneginf`, `isposinf`
- **Tolerance Checks**: `isclose`, `allclose`
- **Array Comparison**: `array_equal`, `array_equiv`
- **Bitwise**: `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`, `left_shift`, `right_shift`, `popcount`

### Math and Trigonometry
- **Element-wise Ops**: `where`, `clip`, `astype`, `sign` + Halide builtins (`Halide::abs`, `Halide::sqrt`, `Halide::exp`, `Halide::log`, `Halide::pow`, `Halide::floor`, `Halide::ceil`, `Halide::round`)
- **Trigonometric**: `hypot`, `degrees`, `radians` + Halide builtins (`Halide::sin`, `Halide::cos`, `Halide::tan`, `Halide::asin`, `Halide::acos`, `Halide::atan`, `Halide::atan2`)
- **Hyperbolic**: `asinh`, `acosh`, `atanh` + Halide builtins (`Halide::sinh`, `Halide::cosh`, `Halide::tanh`)
- **Extended Math**: `exp2`, `log2`, `log10`, `expm1`, `log1p`, `square`, `cbrt`, `reciprocal`, `sinc`, `heaviside`, `fmod`, `remainder`, `nan_to_num`
- **Polynomials**: `polyval`, `chebyshev_t`, `legendre_p`

### Linear Algebra and Sorting
- **Linear Algebra**: `matmul`, `dot`, `outer`, `matvec`, `trace`, `diag`, `norm`, `frobenius_norm`, `triu`, `tril`, `det2x2`, `det3x3`, `inv2x2`
- **Sorting**: `argmin`, `argmax` (1D and 2D with axis), `bitonic_sort`, `bitonic_argsort`, `searchsorted`
- **Set Operations**: `mark_unique`, `count_unique`, `unique`, `in1d`, `intersect1d`, `union1d`, `setdiff1d`
- **Distance**: `cdist_euclidean`, `cdist_manhattan`, `cosine_similarity`

### Signal Processing
- **FFT**: `fft`, `ifft`, `fft2d`, `ifft2d`, `fftshift`, `power_spectrum`
- **Real FFT**: `rfft`, `irfft`, `rfft2d`, `irfft2d`, `fftfreq`, `rfftfreq`
- **Spectral**: `cross_power_spectrum`, `spectral_centroid`
- **Convolution**: `convolve1d`, `convolve2d`, `convolve2d_separable`, `correlate2d`
- **Convolution Kernels**: `box_kernel`, `gaussian_kernel_1d`, `sobel_x_kernel`, `sobel_y_kernel`, `laplacian_kernel`
- **Window Functions**: `hanning`, `hamming`, `blackman`, `bartlett`, `kaiser`

### Image Processing
- **Interpolation**: `interp1d_uniform`, `resize_bilinear`, `resize_nearest`, `zoom`, `map_coordinates`
- **Morphology**: `dilate`, `erode`, `morph_open`, `morph_close`, `morph_gradient`, `top_hat`
- **Color Spaces**: `rgb_to_gray`, `gray_to_rgb`, `rgb_to_hsv`, `hsv_to_rgb`, `rgb_to_yuv`, `yuv_to_rgb`
- **Histogram**: `histogram_1d`, `cumulative_histogram`, `histogram_equalize`, `apply_lut`, `gamma_correct`
- **Thresholding**: `threshold_binary`, `threshold_trunc`, `threshold_tozero`, `threshold_otsu`, `threshold_adaptive`
- **Gradient**: `gradient_1d`, `gradient_2d`, `laplacian`, `divergence`
- **Stencils**: `stencil_apply`, `jacobi_step`, `heat_diffusion_step`

### Performance
- **Scheduling Helpers**: `auto_tile`, `vectorize`, `parallel`, `full_optimize_2d`

## Building

### Prerequisites
- Visual Studio 2022
- .NET 6.0 SDK (for Sharpmake)
- Git

### Setup

1. Clone the repository with submodules:
```bash
git clone --recursive https://github.com/soufianekhiat/NumHalide.git
cd NumHalide
```

Or if already cloned, initialize submodules:
```bash
git submodule update --init --recursive
```

2. **Option A: CMake** (recommended)

Requires [vcpkg](https://vcpkg.io/) with `VCPKG_ROOT` environment variable set.
```bash
cmake --preset default
cmake --build build --config Release
ctest --preset default
```

3. **Option B: Sharpmake**
```bash
buildsystem\Startup.bat          # Build Sharpmake (first time only)
buildsystem\GenerateProjects.bat  # Generate VS projects
```
Then open `NumHalide_win64.sln`.

## Project Structure

```
NumHalide/
├── src/                    # Header-only library (35 headers)
│   ├── numhalide_all.h     # Umbrella include
│   ├── shape.h             # Shape utilities
│   ├── broadcast.h         # Broadcasting
│   ├── factory_func.h      # Array creation
│   ├── manipulation_func.h # Reshape, transpose, slice, flip, rot90, pad
│   ├── reduce.h            # Reductions and boolean reductions
│   ├── stats.h             # Statistics (var, std)
│   ├── la.h                # Linear algebra (matmul, norm, triu/tril, det, inv)
│   ├── ops.h               # Element-wise and comparison operations
│   ├── sort.h              # Sorting (argmin, argmax, bitonic sort)
│   ├── set_ops.h           # Set operations (unique, in1d, intersect, union, diff)
│   ├── conv.h              # Convolution and correlation
│   ├── interp.h            # Interpolation and resampling
│   ├── fft.h               # FFT (1D/2D forward, inverse, power spectrum)
│   ├── schedule.h          # Scheduling helpers
│   ├── trig.h              # Trigonometric functions
│   ├── math_ext.h          # Extended math (exp2, log2, sinc, etc.)
│   ├── cumulative.h        # Cumulative sum, product, diff
│   ├── split.h             # Array splitting
│   ├── compare_ext.h       # isclose, allclose, isneginf, isposinf
│   ├── stats_ext.h         # median, ptp, average, histogram, digitize
│   ├── random_ext.h        # Exponential, Bernoulli, choice distributions
│   ├── array_compare.h     # array_equal, array_equiv
│   ├── bitwise.h           # Bitwise operations and popcount
│   ├── window.h            # Window functions (Hanning, Hamming, etc.)
│   ├── rfft.h              # Real FFT, frequency bins
│   ├── gradient.h          # Gradient, Laplacian, divergence
│   ├── morphology.h        # Dilate, erode, open, close, top-hat
│   ├── color.h             # RGB/HSV/YUV color space conversions
│   ├── polynomial.h        # Polynomial eval, Chebyshev, Legendre
│   ├── distance.h          # Euclidean, Manhattan, cosine distance
│   ├── stencil.h           # Stencil apply, Jacobi, heat diffusion
│   ├── histogram.h         # Histogram, equalization, LUT, gamma
│   ├── threshold.h         # Binary, Otsu, adaptive thresholding
│   └── fft_ext.h           # Cross power spectrum, spectral centroid
├── examples/               # 41 usage examples (each produces a 512x512 PNG)
│   ├── 00_gradient/        # Basic gradient image
│   ├── 01_shape_debug/     # Shape debugging
│   ├── 02_factories/       # Factory functions
│   ├── 03_stacking/        # Concat and stack
│   ├── 04_broadcasting/    # Broadcasting demo
│   ├── 05_reductions/      # Sum, mean, min, max
│   ├── 06_slicing/         # Slice and transpose
│   ├── 07_random/          # Random generation
│   ├── 08_matmul/          # Matrix multiplication
│   ├── 09_masks/           # Where and masking
│   ├── 10_scheduling/      # Performance optimization
│   ├── 11_statistics/      # Variance and standard deviation
│   ├── 12_bool_reduce/     # Boolean reductions
│   ├── 13_manipulation_ext/# Flip, rotate, tile, pad
│   ├── 14_comparisons/     # Comparison and logical operations
│   ├── 15_set_ops/         # Set operations
│   ├── 16_sorting/         # Sorting and search
│   ├── 17_linalg_ext/      # Extended linear algebra
│   ├── 18_fft/             # FFT transforms
│   ├── 19_convolution/     # Filter gallery
│   ├── 20_interpolation/   # Image resizing and warping
│   ├── 21_trigonometry/    # Trigonometric functions
│   ├── 22_math/            # Extended math functions
│   ├── 23_cumulative/      # Cumulative operations
│   ├── 24_splitting/       # Array splitting
│   ├── 25_closeness/       # Tolerance comparisons
│   ├── 26_statistics_ext/  # Extended statistics
│   ├── 27_random_ext/      # Random distributions
│   ├── 28_array_compare/   # Array comparison
│   ├── 29_bitwise/         # Bitwise operations
│   ├── 30_windows/         # Window functions
│   ├── 31_rfft/            # Real FFT
│   ├── 32_gradient/        # Image gradients
│   ├── 33_morphology/      # Morphological operations
│   ├── 34_color/           # Color space conversions
│   ├── 35_polynomial/      # Polynomial evaluation
│   ├── 36_distance/        # Distance computations
│   ├── 37_stencil/         # Stencil / PDE operations
│   ├── 38_histogram/       # Histogram and gamma
│   ├── 39_spectral/        # Spectral analysis
│   └── 40_threshold/       # Thresholding techniques
├── tests/                  # GoogleTest suite
└── buildsystem/            # Build system (Sharpmake configs, scripts, tools)
```

## Quick Start

```cpp
#include "numhalide_all.h"
using namespace numhalide;

// Create arrays
shape_t shape = {4, 4};
auto a = zeros(Halide::Float(32), shape);
auto b = ones(Halide::Float(32), shape);

// Matrix operations
shape_t mat_a = {3, 4};  // 3 rows, 4 cols
shape_t mat_b = {4, 2};  // 4 rows, 2 cols
auto result = la::matmul(a_func, mat_a, b_func, mat_b);  // Result: 3x2

// Reductions
auto sum_all = reduce::sum(a, shape);           // Scalar
auto sum_rows = reduce::sum(a, shape, 0);       // Sum along rows
auto mean_val = reduce::mean(a, shape);         // Mean

// Random arrays
auto uniform = rand_uniform(Halide::Float(32), shape, /*seed=*/42);
auto normal = rand_normal(Halide::Float(32), shape, /*mean=*/0.0f, /*stddev=*/1.0f);

// Scheduling for performance
Halide::Func f = /* your computation */;
schedule::full_optimize_2d(f, 64, 64, 8);  // Tile + vectorize + parallelize
```

## Example: Gaussian Image

```cpp
Func xs = linspace(Float(32), 0.0f, 1.0f, width, "xs");
Func ys = linspace(Float(32), 0.0f, 1.0f, height, "ys");
auto ids = meshgrid(Float(32), { xs, ys }, "meshgrid");
Func x = ids[0], y = ids[1];

Var u, v, c;
Expr cx = 2.0f * (x(u, v) - 0.5f);
Expr cy = 2.0f * (y(u, v) - 0.5f);
Expr out = exp(-(cx * cx + cy * cy) / 0.25f);

// Halide optimizes away intermediate arrays automatically
```

## API Reference

### Factory Functions

| NumPy | NumHalide |
| --- | --- |
| `np.zeros((3, 4))` | `zeros(Float(32), {3, 4})` |
| `np.ones((3, 4))` | `ones(Float(32), {3, 4})` |
| `np.full((3, 4), 5)` | `full(Float(32), {3, 4}, 5.0f)` |
| `np.linspace(0, 10, 5)` | `linspace(Float(32), 0, 10, 5)` |
| `np.arange(3, 7)` | `arange(Float(32), 3, 7)` |
| `np.eye(4)` | `eye(Float(32), 4)` |
| `np.zeros_like(a)` | `zeros_like(a, shape)` |
| `np.ones_like(a)` | `ones_like(a, shape)` |

### Random

| NumPy | NumHalide |
| --- | --- |
| `np.random.rand(3, 4)` | `rand_uniform(Float(32), {3, 4}, seed)` |
| `np.random.randn(3, 4)` | `rand_normal(Float(32), {3, 4})` |
| `np.random.randint(0, 10, (3, 4))` | `rand_int(Int(32), {3, 4}, 0, 10, seed)` |

### Shape Manipulation

| NumPy | NumHalide |
| --- | --- |
| `a.reshape((2, 3))` | `reshape(a, old_shape, {2, 3})` |
| `a.T` / `np.transpose(a)` | `transpose(a, shape, {1, 0})` |
| `np.expand_dims(a, 0)` | `expand_dims(a, shape, 0)` |
| `np.squeeze(a)` | `squeeze(a, shape)` |
| `np.moveaxis(a, 0, 2)` | `moveaxis(a, shape, 0, 2)` |
| `np.flip(a, axis)` | `flip(a, shape, axis)` |
| `np.flipud(a)` | `flipud(a, shape)` |
| `np.fliplr(a)` | `fliplr(a, shape)` |
| `np.rot90(a)` | `rot90(a, shape)` |
| `np.rot90(a, k=2)` | `rot90(a, shape, 2)` |
| `np.roll(a, shift, axis)` | `roll(a, shape, shift, axis)` |
| `np.tile(a, (2, 3))` | `tile(a, shape, {2, 3})` |
| `np.repeat(a, 3, axis)` | `repeat(a, shape, 3, axis)` |
| `np.pad(a, width, mode)` | `pad(a, shape, width, PadMode::Constant)` |

### Slicing

| NumPy | NumHalide |
| --- | --- |
| `a[2:5]` | `slice(a, shape, axis, 2, 5)` |
| `a[::2]` | `slice(a, shape, axis, 0, n, 2)` |
| `a[indices]` | `take(a, shape, indices, axis)` |

### Stacking

| NumPy | NumHalide |
| --- | --- |
| `np.concatenate([a, b], axis=0)` | `concat({a, b}, {shape_a, shape_b}, 0)` |
| `np.stack([a, b], axis=0)` | `stack({a, b}, shape, 0)` |
| `np.vstack([a, b])` | `vstack({a, b}, {shape_a, shape_b})` |
| `np.hstack([a, b])` | `hstack({a, b}, {shape_a, shape_b})` |

### Reductions

| NumPy | NumHalide |
| --- | --- |
| `np.sum(a)` | `reduce::sum(a, shape)` |
| `np.sum(a, axis=0)` | `reduce::sum(a, shape, 0)` |
| `np.mean(a)` | `reduce::mean(a, shape)` |
| `np.min(a)` | `reduce::min(a, shape)` |
| `np.max(a)` | `reduce::max(a, shape)` |
| `np.prod(a)` | `reduce::prod(a, shape)` |

### Statistics

| NumPy | NumHalide |
| --- | --- |
| `np.var(a)` | `var(a, shape)` |
| `np.var(a, axis=0)` | `var(a, shape, 0)` |
| `np.var(a, ddof=1)` | `var(a, shape, 1)` (Bessel correction) |
| `np.std(a)` | `std(a, shape)` |
| `np.std(a, axis=0)` | `std(a, shape, 0)` |

### Boolean Reductions

| NumPy | NumHalide |
| --- | --- |
| `np.any(a)` | `reduce_any(a, shape)` |
| `np.any(a, axis=0)` | `reduce_any(a, shape, 0)` |
| `np.all(a)` | `reduce_all(a, shape)` |
| `np.count_nonzero(a)` | `count_nonzero(a, shape)` |

### Linear Algebra

| NumPy | NumHalide |
| --- | --- |
| `np.matmul(a, b)` / `a @ b` | `matmul(a, shape_a, b, shape_b)` |
| `np.dot(a, b)` | `dot(a, shape_a, b, shape_b)` |
| `np.outer(a, b)` | `outer(a, shape_a, b, shape_b)` |
| `np.trace(a)` | `trace(a, shape)` |
| `np.diag(a)` | `diag(a, shape)` |
| `np.linalg.norm(v)` | `norm(v, shape)` |
| `np.linalg.norm(m, 'fro')` | `frobenius_norm(m, shape)` |
| `np.triu(a)` | `triu(a, shape)` |
| `np.triu(a, k=1)` | `triu(a, shape, 1)` |
| `np.tril(a)` | `tril(a, shape)` |
| `np.linalg.det(a)` (2x2) | `det2x2(a)` |
| `np.linalg.det(a)` (3x3) | `det3x3(a)` |
| `np.linalg.inv(a)` (2x2) | `inv2x2(a)` |

### Comparisons

| NumPy | NumHalide |
| --- | --- |
| `np.equal(a, b)` | `equal(a, b, shape)` |
| `np.not_equal(a, b)` | `not_equal(a, b, shape)` |
| `np.greater(a, b)` | `greater(a, b, shape)` |
| `np.less(a, b)` | `less(a, b, shape)` |
| `np.greater_equal(a, b)` | `greater_equal(a, b, shape)` |
| `np.less_equal(a, b)` | `less_equal(a, b, shape)` |
| `np.logical_and(a, b)` | `logical_and(a, b, shape)` |
| `np.logical_or(a, b)` | `logical_or(a, b, shape)` |
| `np.logical_not(a)` | `logical_not(a, shape)` |
| `np.logical_xor(a, b)` | `logical_xor(a, b, shape)` |
| `np.isnan(a)` | `isnan_func(a, shape)` |
| `np.isinf(a)` | `isinf_func(a, shape)` |
| `np.isfinite(a)` | `isfinite_func(a, shape)` |

### Sorting and Search

| NumPy | NumHalide |
| --- | --- |
| `np.argmin(a)` | `argmin(a, shape)` |
| `np.argmin(a, axis=0)` | `argmin(a, shape, 0)` |
| `np.argmax(a)` | `argmax(a, shape)` |
| `np.argmax(a, axis=1)` | `argmax(a, shape, 1)` |
| `np.sort(a)` | `bitonic_sort(a, size)` (power of 2) |
| `np.argsort(a)` | `bitonic_argsort(a, size)` (power of 2) |
| `np.searchsorted(a, v)` | `searchsorted(a, v, size, n)` |

### Set Operations

| NumPy | NumHalide |
| --- | --- |
| `np.unique(a)` | `unique(a, size)` (sorted input) |
| `np.in1d(a, b)` | `in1d(a, b_sorted, a_size, b_size)` |
| `np.intersect1d(a, b)` | `intersect1d_sorted(a, b, size_a, size_b)` |
| `np.setdiff1d(a, b)` | `setdiff1d_sorted(a, b, size_a, size_b)` |
| `np.union1d(a, b)` | `union1d_sorted(a, b, size_a, size_b)` |

**Helpers:**
- `mark_unique(a, size)` - Mark first occurrence of each value in sorted array
- `count_unique(a, size)` - Count distinct values in sorted array

### FFT

| NumPy | NumHalide |
| --- | --- |
| `np.fft.fft(a)` | `fft(a, N)` |
| `np.fft.ifft(a)` | `ifft(a, N)` |
| `np.fft.fft2(a)` | `fft2d(a, rows, cols)` |
| `np.fft.ifft2(a)` | `ifft2d(a, rows, cols)` |
| `np.fft.fftshift(a)` | `fftshift_1d(a, N)` / `fftshift_2d(a, rows, cols)` |
| `np.abs(np.fft.fft(a))**2` | `power_spectrum(a, N)` / `power_spectrum_2d(a, rows, cols)` |

**Complex number helpers:** `complex()`, `complex_add()`, `complex_mul()`, `complex_conj()`, `complex_mag()`, `expj()`

### Convolution

| NumPy | NumHalide |
| --- | --- |
| `np.convolve(a, k)` | `convolve1d(a, shape, k, k_size)` |
| `scipy.signal.convolve2d(a, k)` | `convolve2d(a, shape, k, k_rows, k_cols)` |
| `scipy.signal.correlate2d(a, k)` | `correlate2d(a, shape, k, k_rows, k_cols)` |
| `scipy.ndimage.convolve(a, k_x, k_y)` | `convolve2d_separable(a, shape, k_x, k_y, k_size)` |

**Built-in Kernels:**
- `box_kernel(size)` - Averaging filter
- `gaussian_kernel_1d(size, sigma)` - Gaussian blur
- `sobel_x_kernel()`, `sobel_y_kernel()` - Edge detection
- `laplacian_kernel()` - Laplacian operator

### Interpolation

| NumPy / SciPy | NumHalide |
| --- | --- |
| `np.interp(x_new, x, y)` | `interp1d_uniform(y, shape, scale)` |
| `cv2.resize(img, ..., INTER_LINEAR)` | `resize_bilinear(a, shape, out_h, out_w)` |
| `cv2.resize(img, ..., INTER_NEAREST)` | `resize_nearest(a, shape, out_h, out_w)` |
| `scipy.ndimage.zoom(a, factor)` | `zoom(a, shape, factor)` |
| `scipy.ndimage.map_coordinates(a, coords)` | `map_coordinates(a, shape, coords_x, coords_y)` |

### Element-wise Operations

| NumPy | NumHalide |
| --- | --- |
| `np.where(cond, a, b)` | `where(cond, a, b, shape)` |
| `np.clip(a, lo, hi)` | `clip(a, shape, lo, hi)` |
| `a.astype(np.int32)` | `astype(a, shape, Int(32))` |
| `np.sign(a)` | `sign(a, shape)` |
| `np.abs(a)` | `Halide::abs(a(Halide::_))` |
| `np.sqrt(a)` | `Halide::sqrt(a(Halide::_))` |
| `np.exp(a)` | `Halide::exp(a(Halide::_))` |
| `np.log(a)` | `Halide::log(a(Halide::_))` |
| `np.pow(a, b)` | `Halide::pow(a(Halide::_), b(Halide::_))` |
| `np.floor(a)` | `Halide::floor(a(Halide::_))` |
| `np.ceil(a)` | `Halide::ceil(a(Halide::_))` |
| `np.round(a)` | `Halide::round(a(Halide::_))` |

### Trigonometric Functions

| NumPy | NumHalide |
| --- | --- |
| `np.sin(a)` | `Halide::sin(a(Halide::_))` |
| `np.cos(a)` | `Halide::cos(a(Halide::_))` |
| `np.tan(a)` | `Halide::tan(a(Halide::_))` |
| `np.arcsin(a)` | `Halide::asin(a(Halide::_))` |
| `np.arccos(a)` | `Halide::acos(a(Halide::_))` |
| `np.arctan(a)` | `Halide::atan(a(Halide::_))` |
| `np.arctan2(y, x)` | `Halide::atan2(y(Halide::_), x(Halide::_))` |
| `np.hypot(x, y)` | `hypot(x, y, shape)` |
| `np.sinh(a)` | `Halide::sinh(a(Halide::_))` |
| `np.cosh(a)` | `Halide::cosh(a(Halide::_))` |
| `np.tanh(a)` | `Halide::tanh(a(Halide::_))` |
| `np.arcsinh(a)` | `asinh(a, shape)` |
| `np.arccosh(a)` | `acosh(a, shape)` |
| `np.arctanh(a)` | `atanh(a, shape)` |
| `np.degrees(a)` | `degrees(a, shape)` |
| `np.radians(a)` | `radians(a, shape)` |

### Extended Math

| NumPy | NumHalide |
| --- | --- |
| `np.exp2(a)` | `exp2(a, shape)` |
| `np.log2(a)` | `log2(a, shape)` |
| `np.log10(a)` | `log10(a, shape)` |
| `np.expm1(a)` | `expm1(a, shape)` |
| `np.log1p(a)` | `log1p(a, shape)` |
| `np.square(a)` | `square(a, shape)` |
| `np.cbrt(a)` | `cbrt(a, shape)` |
| `np.reciprocal(a)` | `reciprocal(a, shape)` |
| `np.sinc(a)` | `sinc(a, shape)` |
| `np.heaviside(a, h0)` | `heaviside(a, h0, shape)` |
| `np.fmod(a, b)` | `fmod(a, b, shape)` |
| `np.remainder(a, b)` | `remainder(a, b, shape)` |
| `np.nan_to_num(a)` | `nan_to_num(a, shape)` |

### Cumulative Operations

| NumPy | NumHalide |
| --- | --- |
| `np.cumsum(a)` | `cumsum(a, shape)` |
| `np.cumprod(a)` | `cumprod(a, shape)` |
| `np.diff(a)` | `diff(a, shape)` |
| `np.diff(a, n=2)` | `diff(a, shape, 0, 2)` |

### Array Splitting

| NumPy | NumHalide |
| --- | --- |
| `np.split(a, 4)` | `split(a, shape, axis, 4)` |
| `np.split(a, [2, 5])` | `split_at(a, shape, axis, {2, 5})` |
| `np.hsplit(a, 4)` | `hsplit(a, shape, 4)` |
| `np.vsplit(a, 4)` | `vsplit(a, shape, 4)` |

### Extended Comparisons

| NumPy | NumHalide |
| --- | --- |
| `np.isclose(a, b)` | `isclose(a, b, shape)` |
| `np.allclose(a, b)` | `allclose(a, b, shape)` |
| `np.isneginf(a)` | `isneginf(a, shape)` |
| `np.isposinf(a)` | `isposinf(a, shape)` |
| `np.array_equal(a, b)` | `array_equal(a, b, shape)` |
| `np.array_equiv(a, b)` | `array_equiv(a, b, shape)` |

### Extended Statistics

| NumPy | NumHalide |
| --- | --- |
| `np.median(a)` | `stats::median(a, shape)` |
| `np.ptp(a)` | `stats::ptp(a, shape)` |
| `np.average(a, weights=w)` | `stats::average(a, w, shape)` |
| `np.histogram(a, bins)` | `stats::histogram(a, shape, bins, min, max)` |
| `np.digitize(a, bins)` | `stats::digitize(a, bins, shape, n_bins)` |

### Extended Random

| NumPy | NumHalide |
| --- | --- |
| `np.random.exponential(lam, shape)` | `rand_exponential(type, shape, lambda, seed)` |
| `np.random.binomial(1, p, shape)` | `rand_bernoulli(type, shape, p, seed)` |
| `np.random.choice(n, shape)` | `rand_choice(type, shape, n, seed)` |

### Bitwise Operations

| NumPy | NumHalide |
| --- | --- |
| `np.bitwise_and(a, b)` | `bitwise_and(a, b, shape)` |
| `np.bitwise_or(a, b)` | `bitwise_or(a, b, shape)` |
| `np.bitwise_xor(a, b)` | `bitwise_xor(a, b, shape)` |
| `np.bitwise_not(a)` | `bitwise_not(a, shape)` |
| `np.left_shift(a, n)` | `left_shift(a, shape, n)` |
| `np.right_shift(a, n)` | `right_shift(a, shape, n)` |
| `popcount(a)` | `popcount(a, shape)` |

### Window Functions

| NumPy | NumHalide |
| --- | --- |
| `np.hanning(N)` | `hanning(N)` |
| `np.hamming(N)` | `hamming(N)` |
| `np.blackman(N)` | `blackman(N)` |
| `np.bartlett(N)` | `bartlett(N)` |
| `np.kaiser(N, beta)` | `kaiser(N, beta)` |

### Real FFT

| NumPy | NumHalide |
| --- | --- |
| `np.fft.rfft(a)` | `rfft(a, N)` |
| `np.fft.irfft(a)` | `irfft(a, N)` |
| `np.fft.rfft2(a)` | `rfft2d(a, rows, cols)` |
| `np.fft.irfft2(a)` | `irfft2d(a, rows, cols)` |
| `np.fft.fftfreq(N)` | `fftfreq(N)` |
| `np.fft.rfftfreq(N)` | `rfftfreq(N)` |

### Spectral Analysis

| NumPy / SciPy | NumHalide |
| --- | --- |
| Cross power spectrum | `cross_power_spectrum(a, b, rows, cols)` |
| Spectral centroid | `spectral_centroid(f, N)` |

### Gradient and Differential Operators

| NumPy | NumHalide |
| --- | --- |
| `np.gradient(a, axis=0)` | `gradient_1d(a, shape, axis)` |
| `np.gradient(a)` (2D) | `gradient_2d(a, shape)` |
| Discrete Laplacian | `laplacian(a, shape)` |
| Divergence | `divergence(fx, fy, shape)` |

### Morphological Operations

| SciPy | NumHalide |
| --- | --- |
| `ndimage.maximum_filter(a, k)` | `dilate(a, shape, k)` |
| `ndimage.minimum_filter(a, k)` | `erode(a, shape, k)` |
| `ndimage.morphology.binary_opening` | `morph_open(a, shape, k)` |
| `ndimage.morphology.binary_closing` | `morph_close(a, shape, k)` |
| Morphological gradient | `morph_gradient(a, shape, k)` |
| Top-hat transform | `top_hat(a, shape, k)` |

### Color Space Conversions

| Operation | NumHalide |
| --- | --- |
| RGB to Grayscale (BT.601) | `rgb_to_gray(f, shape)` |
| Grayscale to RGB | `gray_to_rgb(f, shape)` |
| RGB to HSV | `rgb_to_hsv(f, shape)` |
| HSV to RGB | `hsv_to_rgb(f, shape)` |
| RGB to YUV | `rgb_to_yuv(f, shape)` |
| YUV to RGB | `yuv_to_rgb(f, shape)` |

### Polynomial Evaluation

| NumPy | NumHalide |
| --- | --- |
| `np.polyval(coeffs, x)` | `polyval(coeffs, n, x, shape)` |
| `np.polynomial.chebyshev.chebval(x, [0]*n+[1])` | `chebyshev_t(n, x, shape)` |
| `scipy.special.legendre(n)(x)` | `legendre_p(n, x, shape)` |

### Distance Computations

| SciPy | NumHalide |
| --- | --- |
| `scipy.spatial.distance.cdist(a, b, 'euclidean')` | `cdist_euclidean(a, b, n_a, n_b, dim)` |
| `scipy.spatial.distance.cdist(a, b, 'cityblock')` | `cdist_manhattan(a, b, n_a, n_b, dim)` |
| `scipy.spatial.distance.cosine(a, b)` | `cosine_similarity(a, b, shape)` |

### Stencil Operations

| Operation | NumHalide |
| --- | --- |
| Generic weighted stencil | `stencil_apply(f, shape, weights, offsets_x, offsets_y, n)` |
| Jacobi iteration step | `jacobi_step(f, shape)` |
| Heat diffusion step | `heat_diffusion_step(f, shape, dt, alpha)` |

### Histogram and LUT

| Operation | NumHalide |
| --- | --- |
| `np.histogram(a, bins)` | `histogram_1d(a, shape, bins, min, max)` |
| Cumulative histogram | `cumulative_histogram(a, shape, bins, min, max)` |
| Histogram equalization | `histogram_equalize(a, shape, bins)` |
| Apply lookup table | `apply_lut(a, lut, shape)` |
| Gamma correction | `gamma_correct(a, shape, gamma)` |

### Thresholding

| OpenCV | NumHalide |
| --- | --- |
| `cv2.threshold(a, t, 1, THRESH_BINARY)` | `threshold_binary(a, shape, thresh)` |
| `cv2.threshold(a, t, 1, THRESH_TRUNC)` | `threshold_trunc(a, shape, thresh)` |
| `cv2.threshold(a, t, 1, THRESH_TOZERO)` | `threshold_tozero(a, shape, thresh)` |
| `cv2.threshold(a, 0, 1, THRESH_OTSU)` | `threshold_otsu(a, shape, bins)` |
| Adaptive mean threshold | `threshold_adaptive(a, shape, block_size)` |

### Scheduling Helpers

```cpp
// Individual optimizations
schedule::auto_tile(f, 64, 64);           // Cache-friendly tiling
schedule::vectorize(f, 8);                 // SIMD vectorization
schedule::parallel(f, 1);                  // Parallelize outer loop

// Combined optimization
schedule::full_optimize_2d(f, 64, 64, 8);  // All of the above

// Automatic scheduling
schedule::auto_schedule_2d(f);             // Sensible defaults

// Get optimal vector width for target
int vec_width = schedule::get_vector_width();  // 4, 8, or 16
```

## Running Tests

```bash
cd working_dir/release
NumHalide_Tests.exe
```

## Support Development

[<img src="https://c5.patreon.com/external/logo/become_a_patron_button@2x.png" alt="Become a Patron" width="150"/>](https://www.patreon.com/SoufianeKHIAT)

https://www.patreon.com/SoufianeKHIAT

## Acknowledgments

- Built on [Halide](https://halide-lang.org/)
- Build system: [Sharpmake](https://github.com/ubisoft/sharpmake)
- Inspired by [NumCpp](https://github.com/dpilger26/NumCpp)
