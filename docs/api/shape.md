# Shape Manipulation

`#include "manipulation_func.h"` · `#include "split.h"` · `#include "pad_func.h"`

All functions return `Halide::Func` and take the input shape as `shape_t`. Use `infer_*` helpers to compute the output `shape_t`.

## Reshape and Reorder

| Function | NumPy | Notes |
|---|---|---|
| `reshape(f, shape, new_shape)` | `a.reshape(new_shape)` | Total elements must match |
| `transpose(f, shape, axes)` | `np.transpose(a, axes)` | `axes` is explicit; no default 2D swap |
| `expand_dims(f, shape, axis)` | `np.expand_dims(a, axis)` | Inserts size-1 dimension |
| `squeeze(f, shape)` | `np.squeeze(a)` | Removes all size-1 dimensions |
| `moveaxis(f, shape, src, dst)` | `np.moveaxis(a, src, dst)` | Single axis move only |

## Flip and Rotate

| Function | NumPy | Notes |
|---|---|---|
| `flip(f, shape, axis)` | `np.flip(a, axis)` | — |
| `flipud(f, shape)` | `np.flipud(a)` | Flip along axis 0 |
| `fliplr(f, shape)` | `np.fliplr(a)` | Flip along axis 1 |
| `rot90(f, shape, k=1)` | `np.rot90(a, k)` | Counter-clockwise; 2D only |
| `roll(f, shape, shift, axis)` | `np.roll(a, shift, axis)` | Single axis |

## Tile and Repeat

| Function | NumPy | Notes |
|---|---|---|
| `tile(f, shape, reps)` | `np.tile(a, reps)` | `reps` is `shape_t` |
| `repeat(f, shape, count, axis)` | `np.repeat(a, count, axis)` | Uniform repeat count only |

## Padding

| Function | NumPy | Notes |
|---|---|---|
| `pad(f, shape, width, PadMode::Constant)` | `np.pad(a, width, 'constant')` | — |
| `pad(f, shape, width, PadMode::Edge)` | `np.pad(a, width, 'edge')` | — |
| `pad(f, shape, width, PadMode::Reflect)` | `np.pad(a, width, 'reflect')` | — |
| `pad(f, shape, width, PadMode::Wrap)` | `np.pad(a, width, 'wrap')` | — |

`width` is a uniform scalar; per-axis widths are not supported.

## Slicing and Indexing

`#include "manipulation_func.h"`

| Function | NumPy | Notes |
|---|---|---|
| `slice(f, shape, axis, start, stop, step=1)` | `a[start:stop:step]` | Single-axis slice |
| `take(f, shape, indices_f, axis)` | `np.take(a, indices, axis)` | `indices_f` is a `Func` |

## Splitting

`#include "split.h"`

| Function | NumPy | Notes |
|---|---|---|
| `split(f, shape, axis, n_sections)` | `np.split(a, n, axis)` | Returns `vector<Func>` |
| `split_at(f, shape, axis, {i, j, ...})` | `np.split(a, [i, j], axis)` | Split at explicit indices |
| `hsplit(f, shape, n)` | `np.hsplit(a, n)` | Split along axis 1 |
| `vsplit(f, shape, n)` | `np.vsplit(a, n)` | Split along axis 0 |

## Stacking and Concatenation

`#include "manipulation_func.h"` · `#include "join.h"`

| Function | NumPy | Notes |
|---|---|---|
| `concat({f0, f1}, {s0, s1}, axis)` | `np.concatenate([a, b], axis)` | Shapes may differ along `axis` only |
| `stack({f0, f1}, shape, axis)` | `np.stack([a, b], axis)` | All inputs must have identical shape |
| `vstack({f0, f1}, {s0, s1})` | `np.vstack([a, b])` | Concat along axis 0 |
| `hstack({f0, f1}, {s0, s1})` | `np.hstack([a, b])` | Concat along axis 1 |
| `concat_1d(f0, n0, f1, n1)` | `np.concatenate([a, b])` | 1D fast path |
