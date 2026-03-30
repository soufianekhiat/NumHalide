# Distance and Polynomials

## Distance Computations

`#include "distance.h"`

### Layout Convention

Internal functions use **column-major**: `a(dim, point_idx)`. This is the natural Halide layout.
The `_rm` (row-major) variants accept **row-major**: `a(point_idx, dim)` — matching NumPy / SciPy convention.

| Function | SciPy | Notes |
|---|---|---|
| `cdist_euclidean(a, b, n_a, n_b, dim)` | `cdist(a, b, 'euclidean')` | Column-major input |
| `cdist_euclidean_rm(a, b, n_a, n_b, dim)` | `cdist(a, b, 'euclidean')` | Row-major input |
| `cdist_manhattan(a, b, n_a, n_b, dim)` | `cdist(a, b, 'cityblock')` | Column-major input |
| `cdist_manhattan_rm(a, b, n_a, n_b, dim)` | `cdist(a, b, 'cityblock')` | Row-major input |
| `cosine_similarity(a, b, shape)` | `1 - cosine(a, b)` | Returns similarity (higher = more similar) |

Output shape for cdist: `{n_a, n_b}` where `out(i, j)` = distance between point `i` and point `j`.

## Polynomial Evaluation

`#include "polynomial.h"`

| Function | NumPy / SciPy | Notes |
|---|---|---|
| `polyval(coeffs, n, x, shape)` | `np.polyval(coeffs, x)` | `coeffs` is Func; degree-`n` polynomial; Horner's method |
| `chebyshev_t(n, x, shape)` | `chebval(x, [0]*n+[1])` | Chebyshev T_n via recurrence |
| `legendre_p(n, x, shape)` | `scipy.special.legendre(n)(x)` | Legendre P_n via recurrence |

`polyval` uses Horner's method for numerical stability. `coeffs(0)` = leading coefficient.
