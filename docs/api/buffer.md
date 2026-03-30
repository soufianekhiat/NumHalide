# Buffer Utilities

## Zero-copy Views

`#include "view.h"`

All view operations are O(1): they adjust stride metadata on `Halide::Runtime::Buffer<float>` without copying data. The view shares the same host pointer as the source.

| Function | NumPy | Notes |
|---|---|---|
| `view_transpose(buf)` | `a.T` | 2D only; swaps x/y strides |
| `view_slice(buf, axis, start, n)` | `a[start:start+n, ...]` | Sub-range along one axis |
| `view_reshape(buf, new_dims)` | `a.reshape(new_dims)` | New dims as `vector<int>`; contiguous memory required |

### Func/Shape Bridge

```cpp
// Wrap a Runtime::Buffer as a Halide Func (for use in pipelines)
Halide::Func f = func_from_buffer(buf, "name");

// Extract shape from a buffer's dimension metadata
shape_t sh = shape_from_buffer(buf);
```

`func_from_buffer` supports 1D, 2D, and 3D buffers. Use this when you have a pre-computed `Runtime::Buffer` and want to pass it into a NumHalide pipeline.

## In-place Operations

`#include "inplace.h"`

All functions modify the buffer directly. No new allocation. Work correctly on views produced by `view_transpose` / `view_slice` / `view_reshape`.

| Function | NumPy | Notes |
|---|---|---|
| `inplace_threshold(buf, thresh)` | `a[a < thresh] = thresh` | `buf[i] = max(buf[i], thresh)` |
| `inplace_clamp(buf, lo, hi)` | `np.clip(a, lo, hi, out=a)` | — |
| `inplace_scale(buf, factor)` | `a *= factor` | — |
| `inplace_add_scalar(buf, value)` | `a += value` | — |
| `inplace_exp(buf)` | `np.exp(a, out=a)` | — |
| `inplace_sqrt(buf)` | `np.sqrt(np.maximum(a, 0), out=a)` | Clamps to 0 before sqrt |
| `inplace_gamma(buf, gamma)` | `a[:] = a**gamma` | Clamps input to [0,1] first |
| `inplace_normalize(buf)` | `a = (a-a.min())/(a.max()-a.min())` | Min-max normalization to [0,1] |

All functions optionally take a boolean mask Func as a last argument to apply the operation only where the mask is nonzero.
