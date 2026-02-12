# NumHalide

[NumHalide](https://github.com/soufianekhiat/NumHalide) is a header-only C++20 library providing NumPy-like API built on [Halide](https://halide-lang.org/). It enables expressive array programming with automatic optimization through Halide's compiler.

## Features

- **Factory Functions**: `zeros`, `ones`, `full`, `linspace`, `arange`, `eye`, `meshgrid`
- **Random Generation**: `rand_uniform`, `rand_normal`, `rand_int` with seed control
- **Shape Manipulation**: `reshape`, `transpose`, `expand_dims`, `squeeze`, `moveaxis`, `flip`, `flipud`, `fliplr`, `rot90`, `roll`, `tile`, `repeat`, `pad`
- **Slicing**: `slice`, `take` with NumPy-style indexing
- **Stacking**: `concat`, `stack`, `vstack`, `hstack`
- **Broadcasting**: Automatic shape broadcasting for binary operations
- **Reductions**: `sum`, `mean`, `min`, `max`, `prod` with axis support
- **Statistics**: `var`, `std` with axis support and ddof parameter
- **Boolean Reductions**: `reduce_any`, `reduce_all`, `count_nonzero`
- **Comparisons**: `equal`, `not_equal`, `greater`, `less`, `greater_equal`, `less_equal`
- **Logical Operations**: `logical_and`, `logical_or`, `nh_logical_not`, `logical_xor`
- **Special Value Detection**: `isnan_func`, `isinf_func`, `isfinite_func`
- **Sorting**: `argmin`, `argmax` (1D and 2D with axis), `bitonic_sort`, `bitonic_argsort`, `searchsorted`
- **Set Operations**: `mark_unique`, `count_unique`, `unique`, `in1d`, `intersect1d`, `union1d`, `setdiff1d`
- **Linear Algebra**: `matmul`, `dot`, `outer`, `matvec`, `trace`, `diag`, `norm`, `frobenius_norm`, `triu`, `tril`, `det2x2`, `det3x3`, `inv2x2`
- **Convolution**: `convolve1d`, `convolve2d`, `convolve2d_separable`, `correlate2d`
- **Convolution Kernels**: `box_kernel`, `gaussian_kernel_1d`, `sobel_x_kernel`, `sobel_y_kernel`, `laplacian_kernel`
- **Interpolation**: `interp1d_uniform`, `resize_bilinear`, `resize_nearest`, `zoom`, `map_coordinates`
- **FFT**: `fft`, `ifft`, `fft2d`, `ifft2d`, `fftshift`, `power_spectrum`
- **Element-wise Ops**: `where`, `clip`, `astype`, math functions
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

2. Build Sharpmake (first time only):
```bash
Startup.bat
```

3. Generate Visual Studio projects:
```bash
GenerateProjects.bat
```

4. Open the generated solution:
```
NumHalide_win64.sln
```

## Project Structure

```
NumHalide/
├── src/                    # Header-only library
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
│   └── schedule.h          # Scheduling helpers
├── examples/               # Usage examples
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
│   ├── 16_sorting/         # Sorting and search
│   ├── 17_linalg_ext/      # Extended linear algebra
│   ├── 15_set_ops/         # Set operations
│   ├── 18_fft/             # FFT transforms
│   ├── 19_convolution/     # Filter gallery
│   └── 20_interpolation/   # Image resizing and warping
├── tests/                  # GoogleTest suite (205 tests)
└── sharpmake/              # Build system
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
| `np.logical_not(a)` | `nh_logical_not(a, shape)` |
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
| `np.where(cond, a, b)` | `ops::where(cond, a, b, shape)` |
| `np.clip(a, lo, hi)` | `ops::clip(a, shape, lo, hi)` |
| `a.astype(np.int32)` | `ops::astype(a, shape, Int(32))` |
| `np.abs(a)` | `ops::nh_abs(a, shape)` |
| `np.sqrt(a)` | `ops::nh_sqrt(a, shape)` |
| `np.exp(a)` | `ops::nh_exp(a, shape)` |
| `np.log(a)` | `ops::nh_log(a, shape)` |

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
