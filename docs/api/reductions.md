# Reductions

## Standard Reductions

`#include "reduce.h"`

> Functions are in the `numhalide` namespace — no `reduce::` prefix.

| Function | NumPy | Notes |
|---|---|---|
| `sum(f, shape)` | `np.sum(a)` | Full reduction → scalar Func |
| `sum(f, shape, axis, keepdims=false)` | `np.sum(a, axis)` | Multi-axis: pass multiple calls |
| `mean(f, shape)` | `np.mean(a)` | — |
| `mean(f, shape, axis)` | `np.mean(a, axis)` | — |
| `min(f, shape)` | `np.min(a)` | — |
| `min(f, shape, axis)` | `np.min(a, axis)` | — |
| `max(f, shape)` | `np.max(a)` | — |
| `max(f, shape, axis)` | `np.max(a, axis)` | — |
| `prod(f, shape)` | `np.prod(a)` | — |
| `prod(f, shape, axis)` | `np.prod(a, axis)` | — |
| `argmin(f, shape, axis)` | `np.argmin(a, axis)` | Axis-wise only; for global see `sort.h` |
| `argmax(f, shape, axis)` | `np.argmax(a, axis)` | Axis-wise only |

**`keepdims`:** when `true`, reduced axes remain as size-1 dimensions (same as NumPy `keepdims=True`).

## Boolean Reductions

`#include "reduce.h"`

| Function | NumPy | Notes |
|---|---|---|
| `reduce_any(f, shape)` | `np.any(a)` | Returns 1.0 or 0.0 (float) |
| `reduce_any(f, shape, axis)` | `np.any(a, axis)` | — |
| `reduce_all(f, shape)` | `np.all(a)` | Returns 1.0 or 0.0 (float) |
| `reduce_all(f, shape, axis)` | `np.all(a, axis)` | — |
| `count_nonzero(f, shape)` | `np.count_nonzero(a)` | Returns float |

## NaN-safe Reductions

`#include "nan_ops.h"`

| Function | NumPy | Notes |
|---|---|---|
| `nansum(f, shape)` | `np.nansum(a)` | — |
| `nansum(f, shape, axes, keepdims)` | `np.nansum(a, axis)` | `axes` is `vector<int>` |
| `nanmean(f, shape)` | `np.nanmean(a)` | — |
| `nanmin(f, shape)` | `np.nanmin(a)` | — |
| `nanmax(f, shape)` | `np.nanmax(a)` | — |
| `nanprod(f, shape)` | `np.nanprod(a)` | — |
| `nanstd(f, shape)` | `np.nanstd(a)` | — |
| `nanvar(f, shape)` | `np.nanvar(a)` | — |

NaN elements are treated as the identity value for the operation (0 for sum, 1 for prod, etc.).

## Cumulative Operations

`#include "cumulative.h"`

| Function | NumPy | Notes |
|---|---|---|
| `cumsum(f, shape)` | `np.cumsum(a)` | 1D only |
| `cumprod(f, shape)` | `np.cumprod(a)` | 1D only |
| `diff(f, shape, axis=0, n=1)` | `np.diff(a, n, axis)` | `n`-th order finite difference |

## Shape Inference

```cpp
shape_t out = infer_sum(shape, axis, keepdims);
shape_t out = infer_mean(shape, axis, keepdims);
// Same pattern for min, max, prod
```
