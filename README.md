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
- **Broadcasting**: Automatic shape broadcasting for binary operations; `infer_add/sub/mul/div/pow/minimum/maximum` shape helpers

### Reductions and Statistics
- **Reductions**: `sum`, `mean`, `min`, `max`, `prod` with axis support
- **NaN-safe Reductions**: `nansum`, `nanmean`, `nanmin`, `nanmax`, `nanstd`, `nanvar`, `nanprod`
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
- **Numeric Utilities**: `logaddexp`, `logaddexp2`, `copysign`, `signbit`, `trapz_1d`, `i0` (Bessel I₀), `correlate1d`
- **Polynomials**: `polyval`, `chebyshev_t`, `legendre_p`

### Linear Algebra
- **Core**: `matmul`, `dot`, `outer`, `matvec`, `trace`, `diag`, `norm`, `frobenius_norm`, `triu`, `tril`, `det2x2`, `det3x3`, `inv2x2`
- **Decompositions**: `cholesky`, `qr_gs` (Gram-Schmidt), `svd_jacobi` — up to 8×8 native
- **Large Matrices**: `cholesky_large`, `qr_large`, `svd_large` — validated wrappers up to 32×32
- **Batched**: `batched_cholesky`, `batched_qr`, `batched_svd` — decompose a stack of matrices in one call
- **Einstein Summation**: `einsum("ij,jk->ik", A, B)`, `einsum1("ij->i", A)`; implicit output notation supported

### Sorting and Search
- **Global argmin/argmax**: `global_argmin(a, shape)`, `global_argmax(a, shape)` — returns flat index
- **Axis-wise argmin/argmax**: `argmin(a, shape, axis)`, `argmax(a, shape, axis)` — in `reduce.h`
- **Sorting**: `bitonic_sort`, `bitonic_argsort` (power-of-2 sizes)
- **Soft / Differentiable Sort**: `soft_rank`, `soft_sort`, `soft_argsort` — smooth sigmoid-based approximations, temperature-controlled sharpness
- **Search**: `searchsorted`
- **Set Operations**: `mark_unique`, `count_unique`, `unique`, `in1d`, `intersect1d`, `setdiff1d`, `union1d`
- **Set Helpers**: `is_power_of_2(n)`, `next_power_of_2(n)`

### Signal Processing
- **FFT**: `fft`, `ifft`, `fft2d`, `ifft2d`, `fftshift`, `power_spectrum`
- **Fast FFT**: `fft_fast`, `ifft_fast` — optimized Cooley-Tukey path with bit-reversal precompute
- **Real FFT**: `rfft`, `irfft`, `rfft2d`, `irfft2d`, `fftfreq`, `rfftfreq`
- **Spectral**: `cross_power_spectrum`, `spectral_centroid`
- **Convolution**: `convolve1d` (same/valid/full modes), `convolve2d`, `convolve2d_separable`, `correlate2d`
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

### Complex Numbers
- **Value Type**: `complex_f32` — arithmetic (`+`, `-`, `*`, `/`), `abs()`, `phase()`, `conj()`; polar construction via `complex_from_polar`
- **Buffer**: `ComplexBuffer` — paired re/im `Halide::Buffer<float>` with element access
- **Buffer Ops**: `complex_buf_mul`, `complex_buf_add`, `complex_buf_abs`, `complex_buf_phase`
- **Bridges**: `complex_buf_to_func`, `complex_func_to_buf` — connect `ComplexBuffer` to Halide pipelines

### Buffer Utilities
- **Zero-copy views**: `view_transpose`, `view_slice`, `view_reshape` — O(1) stride manipulation, no data copies
- **In-place ops**: `inplace_threshold`, `inplace_clamp`, `inplace_scale`, `inplace_add_scalar`, `inplace_exp`, `inplace_sqrt`, `inplace_gamma`, `inplace_normalize`
- **Func bridges**: `func_from_buffer(buf, name)` — wrap a `Runtime::Buffer` as a Func; `shape_from_buffer(buf)` — extract shape

### Automatic Differentiation
- **Scalar AD** (`autodiff.h`): `DVar` — differentiable scalar with reverse-mode tape. `dtape_reset()` before each computation. Free functions: `dexp`, `dlog`, `dsin`, `dcos`, `dsqrt`, `dpow`, `dtanh`, `dabs`.
- **Tensor AD** (`autodiff_tensor.h`): `Tensor` — N-dimensional float array (row-major, up to 8D); `TVar` — differentiable tensor with reverse-mode tape. `ttape_reset()` before each computation.
  - Tensor ops: `matmul`, `transpose`, `batch_matmul`, `sum_axis`, `norm`, `trace`, `dot`
  - TVar free functions: `tmatmul`, `tbatch_matmul`, `tdot`, `tsum`, `tmean`, `ttranspose`, `treshape`, `texp`, `tlog`, `tsin`, `tcos`, `tsqrt`, `tpow`, `ttanh`, `tabs`, `trelu`, `tsigmoid`, `tnorm`, `tnormalize`, `ttrace`, `tfrobenius_sq`, `tfrobenius`

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
cmake --build build --config RelWithDebInfo
ctest --preset default
```

> **Note:** The pre-built `Halide.dll` uses a mixed CRT (Release C++ stdlib + Debug C runtime). CMake is configured accordingly — use `RelWithDebInfo` rather than plain `Release` to avoid CRT mismatches.

3. **Option B: Sharpmake**
```bash
buildsystem\Startup.bat          # Build Sharpmake (first time only)
buildsystem\GenerateProjects.bat  # Generate VS projects
```
Then open `NumHalide_win64.sln`.

## Project Structure

```
NumHalide/
├── src/                        # Header-only library (54 headers)
│   ├── numhalide_all.h         # Umbrella include
│   ├── numhalide.h             # Namespace macros
│   ├── shape.h                 # shape_t, nh_require error macro
│   ├── broadcast.h             # Broadcasting + infer_* shape helpers
│   ├── factory_func.h          # zeros, ones, full, linspace, arange, eye, meshgrid
│   ├── manipulation_func.h     # reshape, transpose, expand_dims, flip, rot90, pad, tile, repeat
│   ├── flip_roll.h             # flip, roll variants
│   ├── pad_func.h              # pad with PadMode
│   ├── reduce.h                # sum, mean, min, max, prod, argmin, argmax (with axis)
│   ├── stats.h                 # var, std (namespace stats::)
│   ├── stats_ext.h             # median, ptp, average, histogram, digitize (namespace stats::)
│   ├── statistics2.h           # Additional statistics
│   ├── nan_ops.h               # nansum, nanmean, nanmin, nanmax, nanstd, nanvar, nanprod
│   ├── la.h                    # matmul, dot, outer, norm, triu, tril, det, inv, cholesky, qr, svd
│   ├── la_large.h              # cholesky_large, qr_large, svd_large (up to 32×32)
│   ├── la_batched.h            # batched_cholesky, batched_qr, batched_svd
│   ├── einsum.h                # einsum, einsum1; implicit output notation
│   ├── ops.h                   # where, clip, astype, sign + element-wise ops
│   ├── sort.h                  # global_argmin, global_argmax, bitonic_sort, searchsorted
│   ├── soft_sort.h             # soft_rank, soft_sort, soft_argsort (differentiable)
│   ├── set_ops.h               # unique, in1d, intersect1d, union1d, setdiff1d
│   ├── conv.h                  # convolve1d (same/valid/full), convolve2d, correlate2d
│   ├── interp.h                # interp1d_uniform, resize_bilinear, resize_nearest, zoom
│   ├── fft.h                   # fft, ifft, fft2d, ifft2d, fftshift, power_spectrum
│   ├── fft_fast.h              # fft_fast, ifft_fast — optimized Cooley-Tukey
│   ├── fft_ext.h               # cross_power_spectrum, spectral_centroid
│   ├── rfft.h                  # rfft, irfft, rfft2d, irfft2d, fftfreq, rfftfreq
│   ├── window.h                # hanning, hamming, blackman, bartlett, kaiser
│   ├── gradient.h              # gradient_1d, gradient_2d, laplacian, divergence
│   ├── morphology.h            # dilate, erode, morph_open, morph_close, top_hat
│   ├── color.h                 # rgb_to_gray, rgb_to_hsv, hsv_to_rgb, rgb_to_yuv, yuv_to_rgb
│   ├── polynomial.h            # polyval, chebyshev_t, legendre_p
│   ├── distance.h              # cdist_euclidean, cdist_manhattan, cdist_*_rm, cosine_similarity
│   ├── stencil.h               # stencil_apply, jacobi_step, heat_diffusion_step
│   ├── histogram.h             # histogram_1d, cumulative_histogram, histogram_equalize, apply_lut
│   ├── threshold.h             # threshold_binary, threshold_otsu, threshold_adaptive
│   ├── complex_type.h          # complex_f32, ComplexBuffer, complex_buf_* ops, bridges
│   ├── view.h                  # view_transpose, view_slice, view_reshape; func_from_buffer
│   ├── inplace.h               # inplace_threshold, inplace_clamp, inplace_scale, inplace_exp, ...
│   ├── autodiff.h              # DVar, dtape_reset; dexp, dlog, dsin, dcos, dsqrt, dpow, dtanh
│   ├── autodiff_tensor.h       # Tensor, TVar, ttape_reset; full ND reverse-mode AD
│   ├── schedule.h              # auto_tile, vectorize, parallel, full_optimize_2d
│   ├── trig.h                  # hypot, degrees, radians, asinh, acosh, atanh
│   ├── math_ext.h              # exp2, log2, log10, expm1, log1p, square, cbrt, sinc, nan_to_num
│   ├── numeric.h               # logaddexp, logaddexp2, copysign, trapz_1d, i0, correlate1d
│   ├── cumulative.h            # cumsum, cumprod, diff
│   ├── split.h                 # split, split_at, hsplit, vsplit
│   ├── join.h                  # concat_1d and higher-dim join helpers
│   ├── compare_ext.h           # isclose, allclose, isneginf, isposinf
│   ├── array_compare.h         # array_equal, array_equiv
│   ├── bitwise.h               # bitwise_and/or/xor/not, left_shift, right_shift, popcount
│   ├── random_ext.h            # rand_exponential, rand_bernoulli, rand_choice
│   └── common.h                # Shared internal utilities
├── examples/                   # 54 usage examples (each produces a 512×512 PNG)
│   ├── 00_gradient/            # Basic gradient image
│   ├── 01_shape_debug/         # Shape debugging
│   ├── 02_factories/           # Factory functions
│   ├── 03_stacking/            # Concat and stack
│   ├── 04_broadcasting/        # Broadcasting demo
│   ├── 05_reductions/          # sum, mean, min, max
│   ├── 06_slicing/             # Slice and transpose
│   ├── 07_random/              # Random generation
│   ├── 08_matmul/              # Matrix multiplication
│   ├── 09_masks/               # where and masking
│   ├── 10_scheduling/          # Performance optimization
│   ├── 11_statistics/          # var and std
│   ├── 12_bool_reduce/         # Boolean reductions
│   ├── 13_manipulation_ext/    # flip, rotate, tile, pad
│   ├── 14_comparisons/         # Comparison and logical ops
│   ├── 15_set_ops/             # Set operations
│   ├── 16_sorting/             # Sorting and search
│   ├── 17_linalg_ext/          # Extended linear algebra
│   ├── 18_fft/                 # FFT transforms
│   ├── 19_convolution/         # Filter gallery
│   ├── 20_interpolation/       # Image resizing and warping
│   ├── 21_trigonometry/        # Trig functions
│   ├── 22_math/                # Extended math
│   ├── 23_cumulative/          # Cumulative operations
│   ├── 24_splitting/           # Array splitting
│   ├── 25_closeness/           # Tolerance comparisons
│   ├── 26_statistics_ext/      # Extended statistics
│   ├── 27_random_ext/          # Random distributions
│   ├── 28_array_compare/       # Array comparison
│   ├── 29_bitwise/             # Bitwise operations
│   ├── 30_windows/             # Window functions
│   ├── 31_rfft/                # Real FFT
│   ├── 32_gradient/            # Image gradients
│   ├── 33_morphology/          # Morphological operations
│   ├── 34_color/               # Color space conversions
│   ├── 35_polynomial/          # Polynomial evaluation
│   ├── 36_distance/            # Distance computations
│   ├── 37_stencil/             # Stencil / PDE operations
│   ├── 38_histogram/           # Histogram and gamma
│   ├── 39_spectral/            # Spectral analysis
│   ├── 40_threshold/           # Thresholding
│   ├── 41_padding/             # Padding modes
│   ├── 42_linalg_advanced/     # Cholesky, QR, SVD
│   ├── 43_redsvd/              # Randomized SVD
│   ├── 44_broadcasting/        # Advanced broadcasting patterns
│   ├── 45_sort_fast/           # Bitonic sort and soft sort
│   ├── 46_batched_la/          # Batched linear algebra
│   ├── 48_fft_fast/            # Optimized FFT path
│   ├── 49_multi_reduce/        # Multi-axis reductions
│   ├── 50_views/               # Zero-copy buffer views
│   ├── 51_einsum/              # Einstein summation
│   ├── 52_la_runtime/          # Runtime matrix decompositions
│   ├── 53_inplace/             # In-place buffer operations
│   ├── 54_complex_type/        # Complex number buffers + FFT
│   └── 55_autodiff/            # Automatic differentiation (scalar + tensor)
├── tests/                      # GoogleTest suite (841 tests / 94 suites)
└── buildsystem/                # Build system (Sharpmake configs, scripts, tools)
```

## Quick Start

```cpp
#include "numhalide_all.h"
using namespace numhalide;

// Create arrays
shape_t shape = {4, 4};
auto a = zeros(Halide::Float(32), shape);
auto b = ones(Halide::Float(32), shape);

// Matrix operations (functions are in the numhalide namespace, no la:: prefix)
shape_t mat_a = {3, 4};  // 3 rows, 4 cols
shape_t mat_b = {4, 2};  // 4 rows, 2 cols
auto result = matmul(a_func, mat_a, b_func, mat_b);  // Result: 3×2

// Reductions (functions are in the numhalide namespace, no reduce:: prefix)
auto sum_all  = sum(a, shape);          // Scalar
auto sum_rows = sum(a, shape, 0);       // Sum along axis 0
auto mean_val = mean(a, shape);         // Mean

// Random arrays
auto uniform = rand_uniform(Halide::Float(32), shape, /*seed=*/42);
auto normal  = rand_normal(Halide::Float(32), shape, /*mean=*/0.0f, /*stddev=*/1.0f);

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

Var u, v;
Expr cx = 2.0f * (x(u, v) - 0.5f);
Expr cy = 2.0f * (y(u, v) - 0.5f);
Expr out = exp(-(cx * cx + cy * cy) / 0.25f);

// Halide optimizes away intermediate arrays automatically
```

## Example: Automatic Differentiation

```cpp
// Scalar AD: d/dx [tanh(x) * sin(x)]
dtape_reset();
DVar x(1.2f);
DVar y = dtanh(x) * dsin(x);
y.backward();
float dy_dx = x.grad();

// Tensor AD: gradient of least squares d||Ax - b||² / dx = 2 A^T (Ax - b)
ttape_reset();
TVar A({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}});  // 3×2
TVar x_var(std::vector<float>{1.0f, 1.0f});            // 2-vector
TVar b_var(std::vector<float>{1.0f, 2.0f, 3.0f});      // 3-vector
TVar Ax   = tmatmul(A, x_var);
TVar resid = Ax + (b_var * TVar(-1.0f));
TVar loss  = tdot(resid, resid);
loss.backward();
Tensor grad_x = x_var.grad();  // = 2 * A^T * (Ax - b)
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

> Functions live in the `numhalide` namespace (no `reduce::` prefix).

| NumPy | NumHalide |
| --- | --- |
| `np.sum(a)` | `sum(a, shape)` |
| `np.sum(a, axis=0)` | `sum(a, shape, 0)` |
| `np.mean(a)` | `mean(a, shape)` |
| `np.min(a)` | `min(a, shape)` |
| `np.max(a)` | `max(a, shape)` |
| `np.prod(a)` | `prod(a, shape)` |

### NaN-safe Reductions

| NumPy | NumHalide |
| --- | --- |
| `np.nansum(a)` | `nansum(a, shape)` |
| `np.nanmean(a)` | `nanmean(a, shape)` |
| `np.nanmin(a)` | `nanmin(a, shape)` |
| `np.nanmax(a)` | `nanmax(a, shape)` |
| `np.nanstd(a)` | `nanstd(a, shape)` |
| `np.nanvar(a)` | `nanvar(a, shape)` |
| `np.nanprod(a)` | `nanprod(a, shape)` |

### Statistics

| NumPy | NumHalide |
| --- | --- |
| `np.var(a)` | `stats::var(a, shape)` |
| `np.var(a, axis=0)` | `stats::var(a, shape, 0)` |
| `np.var(a, ddof=1)` | `stats::var(a, shape, 1)` (Bessel correction) |
| `np.std(a)` | `stats::std(a, shape)` |
| `np.std(a, axis=0)` | `stats::std(a, shape, 0)` |

### Boolean Reductions

| NumPy | NumHalide |
| --- | --- |
| `np.any(a)` | `reduce_any(a, shape)` |
| `np.any(a, axis=0)` | `reduce_any(a, shape, 0)` |
| `np.all(a)` | `reduce_all(a, shape)` |
| `np.count_nonzero(a)` | `count_nonzero(a, shape)` |

### Linear Algebra

> Functions live in the `numhalide` namespace (no `la::` prefix).

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
| `np.linalg.det(a)` (2×2) | `det2x2(a)` |
| `np.linalg.det(a)` (3×3) | `det3x3(a)` |
| `np.linalg.inv(a)` (2×2) | `inv2x2(a)` |
| `np.linalg.cholesky(a)` | `cholesky(a, n)` |
| `np.linalg.qr(a)` | `qr_gs(a, m, n)` |
| `np.linalg.svd(a)` | `svd_jacobi(a, n)` |

### Large Matrix Decompositions (up to 32×32)

```cpp
// Validated wrappers — nh_require checks n ≤ 32
auto L   = cholesky_large(a, n);          // L such that L L^T = A
auto qr  = qr_large(a, m, n);            // {Q, R}
auto svd = svd_large(a, n, sweeps);      // {U, S, V}
```

### Batched Linear Algebra

```cpp
// A(col, row, batch_idx) — stack of n×n matrices
auto L_batch   = batched_cholesky(A, n, batch);
auto qr_batch  = batched_qr(A, m, n, batch);     // BatchedQRResult{Q, R}
auto svd_batch = batched_svd(A, n, batch);        // BatchedSVDResult{U, S, V}
```

### Einstein Summation

| Notation | NumHalide |
| --- | --- |
| `np.einsum("ij,jk->ik", A, B)` | `einsum("ij,jk->ik", A, shape_A, B, shape_B)` |
| `np.einsum("ij,jk", A, B)` | `einsum("ij,jk", A, shape_A, B, shape_B)` (implicit output) |
| `np.einsum("ij->i", A)` | `einsum1("ij->i", A, shape_A)` |
| `np.einsum("ii->i", A)` | `einsum1("ii->i", A, shape_A)` (diagonal) |
| `np.einsum("ij->", A)` | `einsum1("ij->", A, shape_A)` (full reduction) |

Implicit output: `"ij,jk"` infers `->ik` (indices appearing in exactly one operand, sorted).

**Shape inference:** `infer_einsum(subscript, shape_A, shape_B)`, `infer_einsum1(subscript, shape_A)`.

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
| `np.argmin(a)` | `global_argmin(a, shape)` — flat index |
| `np.argmin(a, axis=0)` | `argmin(a, shape, 0)` — in `reduce.h` |
| `np.argmax(a)` | `global_argmax(a, shape)` — flat index |
| `np.argmax(a, axis=1)` | `argmax(a, shape, 1)` — in `reduce.h` |
| `np.sort(a)` | `bitonic_sort(a, size)` (size must be power of 2) |
| `np.argsort(a)` | `bitonic_argsort(a, size)` (size must be power of 2) |
| `np.searchsorted(a, v)` | `searchsorted(a, v, size, n)` |

**Soft / differentiable sort** (temperature `tau` controls sharpness; `tau→0` = hard sort):

```cpp
Halide::Func ranks   = soft_rank(f, N, tau);       // Smooth rank [0, N-1]
Halide::Func sorted  = soft_sort(f, N, tau);        // Smooth sorted values
Halide::Func indices = soft_argsort(f, N, tau);     // Smooth permutation
```

### Set Operations

| NumPy | NumHalide |
| --- | --- |
| `np.unique(a)` | `unique(a, size)` (sorted input) |
| `np.in1d(a, b)` | `in1d(a, b_sorted, a_size, b_size)` |
| `np.intersect1d(a, b)` | `intersect1d(a, b, size)` (size must be power of 2) |
| `np.setdiff1d(a, b)` | `setdiff1d(a, b, size)` (size must be power of 2) |
| `np.union1d(a, b)` | `union1d(a, b, size)` (size must be power of 2) |

**Helpers:** `mark_unique(a, size)`, `count_unique(a, size)`, `is_power_of_2(n)`, `next_power_of_2(n)`.

### FFT

| NumPy | NumHalide |
| --- | --- |
| `np.fft.fft(a)` | `fft(a, N)` |
| `np.fft.ifft(a)` | `ifft(a, N)` |
| `np.fft.fft2(a)` | `fft2d(a, rows, cols)` |
| `np.fft.ifft2(a)` | `ifft2d(a, rows, cols)` |
| `np.fft.fftshift(a)` | `fftshift_1d(a, N)` / `fftshift_2d(a, rows, cols)` |
| `np.abs(np.fft.fft(a))**2` | `power_spectrum(a, N)` / `power_spectrum_2d(a, rows, cols)` |

**Optimized path:** `fft_fast(a, N)` / `ifft_fast(a, N)` — Cooley-Tukey with precomputed bit-reversal.

**Complex number helpers:** `complex()`, `complex_add()`, `complex_mul()`, `complex_conj()`, `complex_mag()`, `expj()`.

### Convolution

| NumPy | NumHalide |
| --- | --- |
| `np.convolve(a, k, mode)` | `convolve1d(a, shape, k, k_size, mode)` |
| `scipy.signal.convolve2d(a, k)` | `convolve2d(a, shape, k, k_rows, k_cols)` |
| `scipy.signal.correlate2d(a, k)` | `correlate2d(a, shape, k, k_rows, k_cols)` |
| `scipy.ndimage.convolve(a, kx, ky)` | `convolve2d_separable(a, shape, k_x, k_y, k_size)` |

**Modes for `convolve1d`:** `"same"` (edge-clamp, output size = input size), `"valid"` (no padding, output = n − k + 1), `"full"` (zero-pad, output = n + k − 1).

**Shape inference:** `infer_convolve1d(shape, kernel_size, mode)`.

**Built-in Kernels:** `box_kernel(size)`, `gaussian_kernel_1d(size, sigma)`, `sobel_x_kernel()`, `sobel_y_kernel()`, `laplacian_kernel()`.

### Interpolation

| NumPy / SciPy | NumHalide |
| --- | --- |
| `np.interp(x_new, x, y)` | `interp1d_uniform(y, shape, scale)` |
| `cv2.resize(img, ..., INTER_LINEAR)` | `resize_bilinear(a, shape, out_h, out_w)` |
| `cv2.resize(img, ..., INTER_NEAREST)` | `resize_nearest(a, shape, out_h, out_w)` |
| `scipy.ndimage.zoom(a, factor)` | `zoom(a, shape, factor)` |
| `scipy.ndimage.map_coordinates(a, c)` | `map_coordinates(a, shape, coords_x, coords_y)` |

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
| `np.logaddexp(a, b)` | `logaddexp(a, b, shape)` |
| `np.trapz(y)` | `trapz_1d(y, n)` |
| `scipy.special.i0(a)` | `i0(a, shape)` |

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
| `ndimage.binary_opening` | `morph_open(a, shape, k)` |
| `ndimage.binary_closing` | `morph_close(a, shape, k)` |
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
| Chebyshev T_n | `chebyshev_t(n, x, shape)` |
| Legendre P_n | `legendre_p(n, x, shape)` |

### Distance Computations

> Internal convention: column-major `a(dim, point_idx)`. Use `_rm` variants for row-major `a(point_idx, dim)` (NumPy convention).

| SciPy | NumHalide |
| --- | --- |
| `cdist(a, b, 'euclidean')` | `cdist_euclidean(a, b, n_a, n_b, dim)` |
| `cdist(a, b, 'cityblock')` | `cdist_manhattan(a, b, n_a, n_b, dim)` |
| `cdist(a, b, 'euclidean')` (row-major) | `cdist_euclidean_rm(a, b, n_a, n_b, dim)` |
| `cdist(a, b, 'cityblock')` (row-major) | `cdist_manhattan_rm(a, b, n_a, n_b, dim)` |
| `scipy.spatial.distance.cosine` | `cosine_similarity(a, b, shape)` |

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

### Complex Numbers

```cpp
// Value type
complex_f32 z(3.0f, 4.0f);
float mag   = z.abs();          // 5.0
float phase = z.phase();        // atan2(4, 3)
complex_f32 zc = z.conj();
complex_f32 zp = complex_from_polar(5.0f, 0.927f);

// Buffer (re/im pair)
ComplexBuffer cb(N);
cb.set(i, complex_f32(re, im));
complex_f32 elem = cb(i);

ComplexBuffer product = complex_buf_mul(ca, cb);
auto magnitudes = complex_buf_abs(cb);

// Bridge to Halide pipeline
Halide::Func f = complex_buf_to_func(cb, "signal");
Halide::Func fft_out = fft(f, N, "fft_out");
ComplexBuffer result = complex_func_to_buf(fft_out, N);
```

### Zero-copy Buffer Views

```cpp
// O(1) — no data copied, adjusts strides only
auto transposed = view_transpose(buf_2d);          // swap x/y axes
auto slice      = view_slice(buf, axis, start, n); // sub-range along axis
auto reshaped   = view_reshape(buf, new_dims);     // reinterpret layout

// Bridge Runtime::Buffer ↔ Func
Halide::Func f   = func_from_buffer(buf, "name");
shape_t      sh  = shape_from_buffer(buf);
```

### In-place Buffer Operations

> All functions modify the `Runtime::Buffer<float>` in place (no allocation). Work correctly on views from `view_transpose`/`view_slice`/`view_reshape`.

```cpp
inplace_threshold(buf, thresh);          // buf[i] = max(buf[i], thresh)
inplace_clamp(buf, lo, hi);             // buf[i] = clamp(buf[i], lo, hi)
inplace_scale(buf, factor);             // buf[i] *= factor
inplace_add_scalar(buf, value);         // buf[i] += value
inplace_exp(buf);                       // buf[i] = exp(buf[i])
inplace_sqrt(buf);                      // buf[i] = sqrt(max(buf[i], 0))
inplace_gamma(buf, gamma);             // buf[i] = pow(clamp(buf[i], 0, 1), gamma)
inplace_normalize(buf);                 // buf[i] = (buf[i] - min) / (max - min)
```

### Scalar Automatic Differentiation

Tape-based reverse-mode AD. Pure C++ — no Halide JIT.

```cpp
dtape_reset();
DVar x(1.2f), y(0.5f);
DVar z = dexp(x * x) + dtanh(y);
z.backward();
float dz_dx = x.grad();  // 2x * exp(x²)
float dz_dy = y.grad();  // 1 - tanh²(y)
```

**Available functions:** `dexp`, `dlog`, `dsin`, `dcos`, `dsqrt`, `dpow`, `dtanh`, `dabs`.

### Tensor Automatic Differentiation

Full N-dimensional reverse-mode AD. `Tensor` is row-major; supports scalar, 1D, 2D, and ND (up to 8D).

```cpp
// Tensor class — pure value type, no grad tracking
Tensor A({{1.f, 2.f}, {3.f, 4.f}});  // 2×2 matrix
Tensor b = Tensor::zeros({2});
Tensor c = A.matmul(b);
float  t = A.trace();

// TVar — differentiable wrapper
ttape_reset();
TVar W({{1.f, 0.f}, {0.f, 1.f}});               // leaf
TVar x(std::vector<float>{3.f, 4.f});            // leaf
TVar loss = tnorm(tmatmul(W, x));
loss.backward();
Tensor dW = W.grad();  // ∂‖Wx‖/∂W
Tensor dx = x.grad();  // ∂‖Wx‖/∂x
```

**TVar operators:** `+`, `-`, `*`, `/` (elementwise; scalar broadcast supported).

**Free functions:**

| Category | Functions |
| --- | --- |
| Matrix | `tmatmul`, `ttranspose`, `tbatch_matmul`, `ttrace`, `tdot` |
| Reduction | `tsum`, `tsum(axis)`, `tmean`, `tnorm`, `tfrobenius`, `tfrobenius_sq` |
| Reshape | `treshape`, `tnormalize` |
| Activation | `trelu`, `tsigmoid`, `ttanh` |
| Elementwise | `texp`, `tlog`, `tsin`, `tcos`, `tsqrt`, `tpow`, `tabs` |

**Construction:**
```cpp
TVar s(2.0f);                                   // scalar
TVar v(std::vector<float>{1.f, 2.f, 3.f});      // 1D — use explicit vector, not {}
TVar M({{1.f, 2.f}, {3.f, 4.f}});              // 2D
Tensor nd = Tensor::zeros({2, 3, 4});
TVar T4(nd);                                    // ND
```

> **Note:** `TVar({1.f, 2.f})` is ambiguous between `vector<float>` and `Tensor` constructors. Always use `TVar(std::vector<float>{...})` for 1D leaves.

### Broadcasting Shape Inference

```cpp
// All return the output shape_t for the corresponding op
infer_broadcast(shape_a, shape_b);   // general
infer_add(shape_a, shape_b);
infer_sub(shape_a, shape_b);
infer_mul(shape_a, shape_b);
infer_div(shape_a, shape_b);
infer_minimum(shape_a, shape_b);
infer_maximum(shape_a, shape_b);
// ... equal, not_equal, less, less_equal, greater, greater_equal,
//     logical_and, logical_or, logical_xor
```

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
# From the build output directory
./numhalide_tests.exe

# Run a specific suite
./numhalide_tests.exe --gtest_filter="TensorAD*"
```

841 tests across 94 test suites. Shape-only tests run in < 1 ms; Halide JIT tests typically 30–100 ms each.

## Support Development

[<img src="https://c5.patreon.com/external/logo/become_a_patron_button@2x.png" alt="Become a Patron" width="150"/>](https://www.patreon.com/SoufianeKHIAT)

https://www.patreon.com/SoufianeKHIAT

## Acknowledgments

- Built on [Halide](https://halide-lang.org/)
- Build system: [Sharpmake](https://github.com/ubisoft/sharpmake)
- Inspired by [NumCpp](https://github.com/dpilger26/NumCpp)
