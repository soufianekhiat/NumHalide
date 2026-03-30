# Scheduling

`#include "schedule.h"`

Halide separates *what* to compute (the algorithm) from *how* to compute it (the schedule). NumHalide provides helper wrappers for common scheduling patterns.

These have no NumPy equivalent — they are performance hints specific to the Halide backend.

## Helpers

| Function | Effect |
|---|---|
| `schedule::auto_tile(f, tx, ty)` | Split x/y loops into `tx`×`ty` tiles for cache locality |
| `schedule::vectorize(f, vec_width)` | Vectorize innermost loop (SIMD) |
| `schedule::parallel(f, axis)` | Parallelize the given axis across threads |
| `schedule::full_optimize_2d(f, tx, ty, vec)` | Tile + vectorize + parallelize in one call |
| `schedule::auto_schedule_2d(f)` | Sensible defaults for 2D Funcs |
| `schedule::get_vector_width()` | Returns 4, 8, or 16 depending on target CPU |

## Typical Usage

```cpp
Halide::Func result = /* your computation */;

// Option 1: Combined optimizer
schedule::full_optimize_2d(result, 64, 64, 8);

// Option 2: Manual control
schedule::auto_tile(result, 32, 32);   // tile for cache
schedule::vectorize(result, 8);         // SIMD width
schedule::parallel(result, 1);          // parallelize outer loop

// Realize
Halide::Buffer<float> out = result.realize({width, height});
```

## Scheduling Principles

- **Tile size** (`tx`, `ty`): match your L1 cache footprint. Typically 32–128 for float32.
- **Vector width**: use `schedule::get_vector_width()` for portability, or hardcode 8 for AVX2.
- **Parallelism**: parallelizing axis 1 (rows) is usually most effective for 2D images.
- **compute_root**: call `f.compute_root()` on intermediate Funcs to materialize them and allow downstream reuse.

For compute-heavy pipelines, consider Halide's `auto_schedule` (requires AOT compilation) for optimal tiling and fusion decisions.
