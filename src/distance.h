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

/// @brief Compute pairwise Euclidean distance matrix
/// @param a Input points Func, shape (dim, n_a): a(d, i) for point i, dimension d
/// @param b Input points Func, shape (dim, n_b): b(d, j) for point j, dimension d
/// @param n_a Number of points in a
/// @param n_b Number of points in b
/// @param dim Dimensionality of each point
/// @param name Function name
/// @return Distance matrix Func, shape (n_b, n_a): dist(j, i) = sqrt(sum_d (a(d,i) - b(d,j))^2)
inline
Halide::Func cdist_euclidean(Halide::Func a, Halide::Func b, int n_a, int n_b, int dim, std::string const& name = "cdist_euc")
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
Halide::Func cdist_manhattan(Halide::Func a, Halide::Func b, int n_a, int n_b, int dim, std::string const& name = "cdist_man")
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
	nh_require(nullptr, shape.rank == 1, "cosine_similarity requires 1D vectors");
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

NS_NUM_HALIDE_END
