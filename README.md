# NumHalide

[NumHalide](https://github.com/soufianekhiat/NumHalide) is a header-only C++20 library providing NumPy-like API built on [Halide](https://halide-lang.org/). It enables expressive array programming with automatic optimization through Halide's compiler.

## Features

- **Factory Functions**: `zeros`, `ones`, `full`, `linspace`, `arange`, `eye`, `meshgrid`
- **Random Generation**: `rand_uniform`, `rand_normal`, `rand_int` with seed control
- **Shape Manipulation**: `reshape`, `transpose`, `expand_dims`, `squeeze`, `moveaxis`
- **Slicing**: `slice`, `take` with NumPy-style indexing
- **Stacking**: `concat`, `stack`, `vstack`, `hstack`
- **Broadcasting**: Automatic shape broadcasting for binary operations
- **Reductions**: `sum`, `mean`, `min`, `max`, `prod` with axis support
- **Linear Algebra**: `matmul`, `dot`, `outer`, `matvec`, `trace`, `diag`
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
│   ├── manipulation_func.h # Reshape, transpose, slice
│   ├── reduce.h            # Reductions
│   ├── la.h                # Linear algebra
│   ├── ops.h               # Element-wise operations
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
│   └── 10_scheduling/      # Performance optimization
├── tests/                  # GoogleTest suite (91 tests)
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

### Linear Algebra

| NumPy | NumHalide |
| --- | --- |
| `np.matmul(a, b)` / `a @ b` | `la::matmul(a, shape_a, b, shape_b)` |
| `np.dot(a, b)` | `la::dot(a, shape_a, b, shape_b)` |
| `np.outer(a, b)` | `la::outer(a, shape_a, b, shape_b)` |
| `np.trace(a)` | `la::trace(a, shape)` |
| `np.diag(a)` | `la::diag(a, shape)` |

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
