/// @file distance.h
/// @brief Distance computation operations
///
/// Provides: cdist_euclidean, cdist_manhattan, cosine_similarity

#pragma once

#include "common.h"
#include "shape.h"
#include "reduce.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Pairwise Distance Matrices
// -----------------------------------------------------------------------------

/// @note Layout convention: points are stored COLUMN-MAJOR: a(dimension, point_index).
/// This matches Halide's natural indexing where the fastest-varying dimension comes first.
/// For NumPy-compatible ROW-MAJOR layout (a[point_index, dimension]), use the _rm variants
/// (cdist_euclidean_rm, cdist_manhattan_rm) defined at the bottom of this file.

/// @brief Compute pairwise Euclidean distance matrix
/// @param a Input points Func, shape (dim, n_a): a(d, i) for point i, dimension d
/// @param b Input points Func, shape (dim, n_b): b(d, j) for point j, dimension d
/// @param n_a Number of points in a
/// @param n_b Number of points in b
/// @param dim Dimensionality of each point
/// @param name Function name
/// @return Distance matrix Func, shape (n_b, n_a): dist(j, i) = sqrt(sum_d (a(d,i) - b(d,j))^2)
inline
Halide::Func cdist_euclidean(Halide::Func a, Halide::Func b, int /*n_a*/, int /*n_b*/, int dim, std::string const& name = "cdist_euc")
{
	Halide::Func sq_sum(name + "_sq");
	Halide::Var i, j;
	Halide::RDom d(0, dim);

	Halide::Expr diff = a(d, i) - b(d, j);
	sq_sum(j, i) = Halide::cast<float>(0);
	sq_sum(j, i) += diff * diff;

	// Take sqrt
	Halide::Func ret(name);
	ret(j, i) = Halide::sqrt(sq_sum(j, i));
	return ret;
}

/// @brief Compute pairwise Manhattan (L1) distance matrix
/// @param a Input points Func, shape (dim, n_a): a(d, i) for point i, dimension d
/// @param b Input points Func, shape (dim, n_b): b(d, j) for point j, dimension d
/// @param n_a Number of points in a
/// @param n_b Number of points in b
/// @param dim Dimensionality of each point
/// @param name Function name
/// @return Distance matrix Func, shape (n_b, n_a): dist(j, i) = sum_d |a(d,i) - b(d,j)|
inline
Halide::Func cdist_manhattan(Halide::Func a, Halide::Func b, int /*n_a*/, int /*n_b*/, int dim, std::string const& name = "cdist_man")
{
	Halide::Func ret(name);
	Halide::Var i, j;
	Halide::RDom d(0, dim);

	Halide::Expr diff = a(d, i) - b(d, j);
	ret(j, i) = Halide::cast<float>(0);
	ret(j, i) += Halide::abs(diff);
	return ret;
}

/// @brief Compute cosine similarity between two 1D vectors
/// @param a First vector Func
/// @param b Second vector Func
/// @param shape Shape of the input vectors (must be 1D)
/// @param name Function name
/// @return Scalar Func: dot(a,b) / (norm(a) * norm(b))
inline
Halide::Func cosine_similarity(Halide::Func a, Halide::Func b, const shape_t& shape, std::string const& name = "cosine_sim")
{
	nh_require(shape.rank == 1, "cosine_similarity requires 1D vectors");
	int N = shape.extents[0];

	Halide::Func dot_ab(name + "_dot");
	Halide::Func norm_a(name + "_na");
	Halide::Func norm_b(name + "_nb");
	Halide::Var dummy;
	Halide::RDom r(0, N);

	dot_ab(dummy) = Halide::cast<float>(0);
	dot_ab(dummy) += Halide::cast<float>(a(r)) * Halide::cast<float>(b(r));

	norm_a(dummy) = Halide::cast<float>(0);
	norm_a(dummy) += Halide::cast<float>(a(r)) * Halide::cast<float>(a(r));

	norm_b(dummy) = Halide::cast<float>(0);
	norm_b(dummy) += Halide::cast<float>(b(r)) * Halide::cast<float>(b(r));

	Halide::Func ret(name);
	Halide::Expr denom = Halide::sqrt(norm_a(dummy)) * Halide::sqrt(norm_b(dummy));
	ret(dummy) = dot_ab(dummy) / Halide::max(denom, 1e-10f);
	return ret;
}

// =============================================================================
// Row-major (NumPy-compatible) distance wrappers
// =============================================================================
// The core cdist functions use column-major layout: a(dim, point_idx).
// These wrappers accept row-major (NumPy) layout: a(point_idx, dim) and
// internally transpose before computing distances.

/// @brief Euclidean pairwise distances — NumPy-compatible (row-major) layout.
/// @param a   Func where a(i, d) = d-th feature of i-th point (n_a rows x dim cols)
/// @param b   Func where b(j, d) = d-th feature of j-th point (n_b rows x dim cols)
/// @param n_a Number of points in a
/// @param n_b Number of points in b
/// @param dim Feature dimension
/// @param name Function name
/// @return dist(j, i) = ||a[i] - b[j]||_2
inline Halide::Func cdist_euclidean_rm(
    Halide::Func a, Halide::Func b,
    int n_a, int n_b, int dim,
    const std::string& name = "cdist_eucl_rm")
{
    // Transpose: a_col(d, i) = a(i, d), b_col(d, j) = b(j, d)
    Halide::Func a_col("a_col_" + name), b_col("b_col_" + name);
    Halide::Var d_v("d"), i_v("i"), j_v("j");
    a_col(d_v, i_v) = a(i_v, d_v);
    b_col(d_v, j_v) = b(j_v, d_v);
    return cdist_euclidean(a_col, b_col, n_a, n_b, dim, name);
}

/// @brief Manhattan pairwise distances — NumPy-compatible (row-major) layout.
/// @param a   Func where a(i, d) = d-th feature of i-th point (n_a rows x dim cols)
/// @param b   Func where b(j, d) = d-th feature of j-th point (n_b rows x dim cols)
/// @param n_a Number of points in a
/// @param n_b Number of points in b
/// @param dim Feature dimension
/// @param name Function name
/// @return dist(j, i) = ||a[i] - b[j]||_1
inline Halide::Func cdist_manhattan_rm(
    Halide::Func a, Halide::Func b,
    int n_a, int n_b, int dim,
    const std::string& name = "cdist_manh_rm")
{
    Halide::Func a_col("a_col_" + name), b_col("b_col_" + name);
    Halide::Var d_v("d"), i_v("i"), j_v("j");
    a_col(d_v, i_v) = a(i_v, d_v);
    b_col(d_v, j_v) = b(j_v, d_v);
    return cdist_manhattan(a_col, b_col, n_a, n_b, dim, name);
}

NS_NUM_HALIDE_END
