# Comparisons and Logical Operations

## Element-wise Comparisons

`#include "ops.h"`

All functions return a float Func (1.0 = true, 0.0 = false).

| Function | NumPy | Notes |
|---|---|---|
| `equal(f, g, shape)` | `np.equal(a, b)` | — |
| `not_equal(f, g, shape)` | `np.not_equal(a, b)` | — |
| `greater(f, g, shape)` | `np.greater(a, b)` | — |
| `greater_equal(f, g, shape)` | `np.greater_equal(a, b)` | — |
| `less(f, g, shape)` | `np.less(a, b)` | — |
| `less_equal(f, g, shape)` | `np.less_equal(a, b)` | — |

## Logical Operations

`#include "ops.h"`

| Function | NumPy | Notes |
|---|---|---|
| `logical_and(f, g, shape)` | `np.logical_and(a, b)` | Nonzero = true |
| `logical_or(f, g, shape)` | `np.logical_or(a, b)` | — |
| `logical_not(f, shape)` | `np.logical_not(a)` | — |
| `logical_xor(f, g, shape)` | `np.logical_xor(a, b)` | — |

## Bitwise Operations

`#include "bitwise.h"`

| Function | NumPy | Notes |
|---|---|---|
| `bitwise_and(f, g, shape)` | `np.bitwise_and(a, b)` | Integer types |
| `bitwise_or(f, g, shape)` | `np.bitwise_or(a, b)` | — |
| `bitwise_xor(f, g, shape)` | `np.bitwise_xor(a, b)` | — |
| `bitwise_not(f, shape)` | `np.bitwise_not(a)` | — |
| `left_shift(f, shape, n)` | `np.left_shift(a, n)` | Scalar shift only |
| `right_shift(f, shape, n)` | `np.right_shift(a, n)` | Scalar shift only |
| `popcount(f, shape)` | *(no direct equivalent)* | Count set bits per element |

## Special Value Tests

`#include "ops.h"` · `#include "compare_ext.h"`

| Function | NumPy | Notes |
|---|---|---|
| `isnan_func(f, shape)` | `np.isnan(a)` | Returns float 0/1 |
| `isinf_func(f, shape)` | `np.isinf(a)` | — |
| `isfinite_func(f, shape)` | `np.isfinite(a)` | — |
| `isneginf(f, shape)` | `np.isneginf(a)` | — |
| `isposinf(f, shape)` | `np.isposinf(a)` | — |

## Tolerance and Equality

`#include "compare_ext.h"` · `#include "array_compare.h"`

| Function | NumPy | Notes |
|---|---|---|
| `isclose(f, g, shape, rtol, atol)` | `np.isclose(a, b, rtol, atol)` | Elementwise; returns float 0/1 |
| `allclose(f, g, shape, rtol, atol)` | `np.allclose(a, b, rtol, atol)` | Scalar bool (realized immediately) |
| `array_equal(f, g, shape)` | `np.array_equal(a, b)` | Exact equality; scalar bool |
| `array_equiv(f, g, shape_f, shape_g)` | `np.array_equiv(a, b)` | Equal after broadcasting |
