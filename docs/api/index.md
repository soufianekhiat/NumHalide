# NumHalide API Reference

All functions live in the `numhalide` namespace. Include `numhalide_all.h`.

| File | Categories |
|---|---|
| [array_creation.md](array_creation.md) | Factory functions, random generation |
| [shape.md](shape.md) | Reshape, transpose, slice, split, stack, pad |
| [reductions.md](reductions.md) | sum/mean/min/max, NaN-safe, boolean, cumulative |
| [statistics.md](statistics.md) | var, std, median, histogram, digitize |
| [math.md](math.md) | Elementwise, trigonometry, extended math, numeric |
| [comparisons.md](comparisons.md) | Comparisons, logical, NaN/Inf tests, tolerance |
| [linear_algebra.md](linear_algebra.md) | matmul, norm, decompositions, batched LA, einsum |
| [sorting.md](sorting.md) | argmin/argmax, bitonic sort, soft sort, set ops |
| [signal.md](signal.md) | FFT, real FFT, windows, spectral, convolution |
| [image.md](image.md) | Interpolation, morphology, color, histogram, threshold, gradient |
| [distance.md](distance.md) | Euclidean/Manhattan cdist, cosine similarity, polynomials |
| [complex.md](complex.md) | complex_f32, ComplexBuffer, Halide bridge |
| [buffer.md](buffer.md) | Zero-copy views, in-place ops, Func bridge |
| [autodiff.md](autodiff.md) | Scalar AD (DVar), tensor AD (Tensor / TVar) |
| [broadcasting.md](broadcasting.md) | Broadcast rules, infer_* shape helpers |
| [scheduling.md](scheduling.md) | Tiling, vectorization, parallelism |

## Key differences from NumPy

- **Lazy evaluation**: functions return `Halide::Func`, not arrays. Call `.realize({...})` to materialize.
- **Explicit shapes**: most functions take `shape_t` alongside the `Func`. Use `infer_*` helpers to compute output shapes.
- **Explicit types**: factory functions take `Halide::Type` (e.g. `Float(32)`, `Int(32)`).
- **Namespace**: `stats::var`, `stats::std`, `stats::median` live in a nested `stats::` namespace. Everything else is flat in `numhalide`.
- **Column-major internal convention**: Halide buffers are column-major (`x` is innermost). All NumHalide functions follow Halide convention internally. Row-major wrappers are provided where noted.
