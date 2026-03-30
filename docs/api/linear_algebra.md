# Linear Algebra

## Core Operations

`#include "la.h"` — functions in `numhalide` namespace (no `la::` prefix).

| Function | NumPy | Notes |
|---|---|---|
| `matmul(a, sa, b, sb)` | `a @ b` / `np.matmul(a, b)` | Takes shape alongside each Func |
| `dot(a, sa, b, sb)` | `np.dot(a, b)` | 1D dot product |
| `outer(a, sa, b, sb)` | `np.outer(a, b)` | 1D inputs → 2D output |
| `matvec(A, sA, x, sx)` | `A @ x` | Matrix × vector shorthand |
| `trace(f, shape)` | `np.trace(a)` | Sum of diagonal elements |
| `diag(f, shape)` | `np.diag(a)` | 1D→diagonal 2D, or 2D→diagonal 1D |
| `norm(f, shape)` | `np.linalg.norm(a)` | L2/Frobenius depending on rank |
| `frobenius_norm(f, shape)` | `np.linalg.norm(a, 'fro')` | Explicit Frobenius |
| `triu(f, shape, k=0)` | `np.triu(a, k)` | Upper triangle; `k` shifts diagonal |
| `tril(f, shape, k=0)` | `np.tril(a, k)` | Lower triangle |
| `det2x2(f)` | `np.linalg.det(a)` | 2×2 only |
| `det3x3(f)` | `np.linalg.det(a)` | 3×3 only |
| `inv2x2(f)` | `np.linalg.inv(a)` | 2×2 only |
| `batched_matmul(A, sa, B, sb)` | `np.matmul(A, B)` | 3D batched: `(batch, m, k) @ (batch, k, n)` |

**Shape inference:**
```cpp
shape_t sc = infer_matmul(sa, sb);
shape_t so = infer_outer(sa, sb);
shape_t sb = infer_batched_matmul(sa, sb);
```

## Matrix Decompositions

`#include "la.h"`

| Function | NumPy | Notes |
|---|---|---|
| `cholesky(A, n)` | `np.linalg.cholesky(a)` | Up to 8×8; A must be symmetric PD |
| `qr_gs(A, m, n)` | `np.linalg.qr(a)` | Gram-Schmidt; returns `{Q, R}` |
| `svd_jacobi(A, n, sweeps)` | `np.linalg.svd(a)` | Jacobi iterations; up to 8×8; returns `{U, S, V}` |

## Large Matrix Decompositions (up to 32×32)

`#include "la_large.h"`

Validated wrappers: raise `std::runtime_error` if `n > 32`.

| Function | NumPy | Notes |
|---|---|---|
| `cholesky_large(A, n)` | `np.linalg.cholesky(a)` | `n ≤ 32`; nh_require enforced |
| `qr_large(A, m, n)` | `np.linalg.qr(a)` | `n ≤ 32`; returns `{Q, R}` |
| `svd_large(A, n, sweeps)` | `np.linalg.svd(a)` | `n ≤ 32`; SVD_n8 takes ~6 min — avoid n > 8 |

> **SVD performance:** `svd_jacobi`/`svd_large` compile time scales as O(n⁴). n=8 takes ~6 min; n=16 would take ~14 hours. For large n use randomized methods.

## Batched Linear Algebra

`#include "la_batched.h"`

Input convention: `A(col, row, batch_idx)` — column-major 2D per batch slice.

| Function | NumPy | Notes |
|---|---|---|
| `batched_cholesky(A, n, batch)` | `np.linalg.cholesky(A)` | Returns `Func` with batch dim |
| `batched_qr(A, m, n, batch)` | `np.linalg.qr(A)` | Returns `BatchedQRResult{Q, R}` |
| `batched_svd(A, n, batch, sweeps)` | `np.linalg.svd(A)` | Returns `BatchedSVDResult{U, S, V}` |

## Einstein Summation

`#include "einsum.h"`

| Function | NumPy | Notes |
|---|---|---|
| `einsum(subscript, A, sA, B, sB)` | `np.einsum(subscript, a, b)` | Implicit output supported |
| `einsum1(subscript, A, sA)` | `np.einsum(subscript, a)` | Single-input (trace, diagonal, reduce) |
| `infer_einsum(subscript, sA, sB)` | — | Output shape |
| `infer_einsum1(subscript, sA)` | — | Output shape |

**Subscript examples:**

| Subscript | Operation |
|---|---|
| `"ij,jk->ik"` | Matrix multiply |
| `"ij,jk"` | Matrix multiply (implicit output — same result) |
| `"ij->i"` | Row sum |
| `"ii->i"` | Diagonal extraction |
| `"ij->"` | Full sum |
| `"i,i->"` | Dot product |
| `"i,j->ij"` | Outer product |

Implicit output: indices appearing in exactly one operand, sorted alphabetically. `"ij,jk"` → `"ij,jk->ik"`.
