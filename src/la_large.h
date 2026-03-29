/// @file la_large.h
/// @brief Extensions to la.h for larger matrix support (up to LA_LARGE_MAX_N x LA_LARGE_MAX_N)
///
/// Provides validated wrappers around the core la.h decompositions
/// (cholesky, qr_gs, svd_jacobi) for matrices up to 32x32, along with
/// appropriate default sweep counts for SVD convergence at larger sizes.

#pragma once
#include "la.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Size limit
// -----------------------------------------------------------------------------

/// Maximum matrix dimension supported by the "large" variants.
/// Beyond this size the Halide stage count becomes impractical for
/// the Jacobi-based algorithms.
static constexpr int LA_LARGE_MAX_N = 32;

// -----------------------------------------------------------------------------
// Cholesky (large)
// -----------------------------------------------------------------------------

/// @brief Cholesky decomposition for matrices up to LA_LARGE_MAX_N x LA_LARGE_MAX_N
///
/// Identical algorithm to cholesky(), but validates n <= LA_LARGE_MAX_N before
/// delegating.  Returns the lower-triangular factor L such that A = L * L^T.
///
/// @param A    Positive-definite n×n matrix Func (A(col, row))
/// @param n    Matrix size (must be in [1, LA_LARGE_MAX_N])
/// @param name Base Func name
/// @return     Lower-triangular Func L(col, row)
inline Halide::Func cholesky_large(Halide::Func A, int n,
    std::string const& name = "chol_large")
{
    nh_require(nullptr, n > 0 && n <= LA_LARGE_MAX_N,
        "cholesky_large: n=%d must be in [1, %d]", n, LA_LARGE_MAX_N);
    return cholesky(A, n, name);
}

/// @brief Infer output shape for cholesky_large: n×n lower-triangular result
inline shape_t infer_cholesky_large(int n) { return shape_t{n, n}; }

// -----------------------------------------------------------------------------
// QR (large)
// -----------------------------------------------------------------------------

/// @brief QR decomposition for matrices up to LA_LARGE_MAX_N columns
///
/// Validates n <= LA_LARGE_MAX_N and m >= n, then delegates to qr_gs().
/// Returns Q (m×n orthonormal) and R (n×n upper-triangular) such that A = Q * R.
///
/// @param A    m×n matrix Func (A(col, row), m >= n)
/// @param m    Number of rows
/// @param n    Number of columns (must be in [1, LA_LARGE_MAX_N])
/// @param name Base Func name
/// @return     QRResult {Q (m×n), R (n×n)}
inline QRResult qr_large(Halide::Func A, int m, int n,
    std::string const& name = "qr_large")
{
    nh_require(nullptr, n > 0 && n <= LA_LARGE_MAX_N,
        "qr_large: n=%d must be in [1, %d]", n, LA_LARGE_MAX_N);
    nh_require(nullptr, m >= n,
        "qr_large: m=%d must be >= n=%d", m, n);
    return qr_gs(A, m, n, name);
}

/// @brief Infer Q output shape for qr_large: m×n orthonormal factor
inline shape_t infer_qr_large_Q(int m, int n) { return shape_t{m, n}; }

/// @brief Infer R output shape for qr_large: n×n upper-triangular factor
inline shape_t infer_qr_large_R(int n) { return shape_t{n, n}; }

// -----------------------------------------------------------------------------
// SVD (large)
// -----------------------------------------------------------------------------

/// @brief Full SVD for matrices up to LA_LARGE_MAX_N columns
///
/// Uses one-sided Jacobi sweeps via svd_jacobi().  When n_sweeps < 0 the sweep
/// count is chosen automatically to balance convergence against compile time:
///
///     n_sweeps = max(3, 5 + n/4)
///
/// For n=4  -> 6 sweeps  (24 pairs  -> ~144 stages)
/// For n=8  -> 7 sweeps  (28 pairs  -> ~196 stages)
/// For n=16 -> 9 sweeps  (36 pairs  -> ~324 stages)
/// For n=32 -> 13 sweeps (52 pairs  -> ~676 stages — acceptable)
///
/// @param A        m×n matrix Func (A(col, row), m >= n)
/// @param m        Number of rows
/// @param n        Number of columns (must be in [1, LA_LARGE_MAX_N])
/// @param n_sweeps Number of Jacobi sweeps (-1 = auto-scale)
/// @param name     Base Func name
/// @return         SVDResult {U (m×n), S (1D size n), Vt (n×n)}
inline SVDResult svd_large(Halide::Func A, int m, int n,
    int n_sweeps = -1,
    std::string const& name = "svd_large")
{
    nh_require(nullptr, n > 0 && n <= LA_LARGE_MAX_N,
        "svd_large: n=%d must be in [1, %d]", n, LA_LARGE_MAX_N);
    nh_require(nullptr, m >= n,
        "svd_large: m=%d must be >= n=%d", m, n);
    if (n_sweeps < 0)
        n_sweeps = std::max(3, 5 + n / 4);  // auto-scale sweep count
    return svd_jacobi(A, m, n, n_sweeps, name);
}

NS_NUM_HALIDE_END
