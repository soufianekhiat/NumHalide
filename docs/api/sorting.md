# Sorting, Search, and Set Operations

## Global Argmin / Argmax

`#include "sort.h"`

| Function | NumPy | Notes |
|---|---|---|
| `global_argmin(f, shape)` | `np.argmin(a)` | Returns flat index; 1D input only |
| `global_argmax(f, shape)` | `np.argmax(a)` | Returns flat index; 1D input only |

For axis-wise argmin/argmax, use `argmin(f, shape, axis)` / `argmax(f, shape, axis)` from `reduce.h`.

## Sorting

`#include "sort.h"`

| Function | NumPy | Notes |
|---|---|---|
| `bitonic_sort(f, size)` | `np.sort(a)` | `size` must be a power of 2 |
| `bitonic_argsort(f, size)` | `np.argsort(a)` | `size` must be a power of 2 |
| `searchsorted(a, v, size, n)` | `np.searchsorted(a, v)` | `a` sorted; `v` is a Func of `n` queries |

**Power-of-2 requirement:** `bitonic_sort` implements a bitonic network. Use `next_power_of_2(n)` (from `set_ops.h`) to pad if needed.

## Soft / Differentiable Sort

`#include "soft_sort.h"`

Smooth approximations via sigmoid-based pairwise comparisons. Temperature `tau` controls sharpness: `tau → 0` recovers hard sort, large `tau` smooths toward uniform.

| Function | NumPy analogue | Notes |
|---|---|---|
| `soft_rank(f, N, tau)` | `np.argsort(np.argsort(a))` | Smooth rank in `[0, N-1]` |
| `soft_sort(f, N, tau)` | `np.sort(a)` | Smooth sorted values |
| `soft_argsort(f, N, tau)` | `np.argsort(a)` | Smooth permutation |

No size restriction. Suitable for gradient-based optimization through sorting.

## Set Operations

`#include "set_ops.h"`

**All inputs must be sorted.** `intersect1d`, `setdiff1d`, and `union1d` require sizes to be powers of 2 (bitonic network constraint); an `nh_require` check raises `std::runtime_error` otherwise.

| Function | NumPy | Notes |
|---|---|---|
| `unique(f, size)` | `np.unique(a)` | Returns (values Func, count Func) |
| `mark_unique(f, size)` | — | 1.0 at first occurrence of each value |
| `count_unique(f, size)` | — | Scalar count of distinct values |
| `in1d(a, b_sorted, a_size, b_size)` | `np.in1d(a, b)` | Returns float mask (0/1) |
| `intersect1d(a, b, size)` | `np.intersect1d(a, b)` | `size` must be power of 2 |
| `setdiff1d(a, b, size)` | `np.setdiff1d(a, b)` | `size` must be power of 2 |
| `union1d(a, b, size)` | `np.union1d(a, b)` | `size` must be power of 2 |

**Helpers:**
```cpp
bool ok   = is_power_of_2(n);      // true if n is a power of 2
int  next = next_power_of_2(n);    // smallest power of 2 ≥ n
```
