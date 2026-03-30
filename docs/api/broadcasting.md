# Broadcasting

`#include "broadcast.h"`

## Rules

NumHalide follows NumPy broadcasting semantics: dimensions are aligned from the trailing axis. A size-1 dimension is stretched to match the other operand.

```
(3, 1) + (1, 4) → (3, 4)
(1, 4) + (4,)   → (1, 4) + (1, 4) → (1, 4)  [1D padded to 2D]
```

The `infer_broadcast` function computes the output shape at build time and throws `std::runtime_error` (via `nh_require`) if shapes are incompatible.

## Shape Inference Helpers

All helpers take two `shape_t` arguments and return the broadcast output `shape_t`.

| Function | Equivalent operation | Notes |
|---|---|---|
| `infer_broadcast(a, b)` | Any binary op | General broadcast shape |
| `infer_add(a, b)` | `a + b` | Alias for `infer_broadcast` |
| `infer_sub(a, b)` | `a - b` | — |
| `infer_mul(a, b)` | `a * b` | — |
| `infer_div(a, b)` | `a / b` | — |
| `infer_pow(a, b)` | `a ** b` | — |
| `infer_minimum(a, b)` | `np.minimum(a, b)` | — |
| `infer_maximum(a, b)` | `np.maximum(a, b)` | — |
| `infer_equal(a, b)` | `np.equal(a, b)` | — |
| `infer_not_equal(a, b)` | `np.not_equal(a, b)` | — |
| `infer_less(a, b)` | `np.less(a, b)` | — |
| `infer_less_equal(a, b)` | `np.less_equal(a, b)` | — |
| `infer_greater(a, b)` | `np.greater(a, b)` | — |
| `infer_greater_equal(a, b)` | `np.greater_equal(a, b)` | — |
| `infer_logical_and(a, b)` | `np.logical_and(a, b)` | — |
| `infer_logical_or(a, b)` | `np.logical_or(a, b)` | — |
| `infer_logical_xor(a, b)` | `np.logical_xor(a, b)` | — |

## Usage Pattern

```cpp
shape_t sa = {3, 1};
shape_t sb = {1, 4};
shape_t sc = infer_add(sa, sb);   // {3, 4}

Halide::Func c = broadcast_add(a, sa, b, sb);
// realize with sc
Halide::Buffer<float> out = c.realize({sc[0], sc[1]});
```

## Difference from NumPy

NumPy broadcasting happens implicitly at runtime. NumHalide requires you to call `infer_broadcast` (or an `infer_*` alias) explicitly to obtain the output shape, then pass that shape to `.realize()`. This is because Halide needs buffer extents at JIT time.
