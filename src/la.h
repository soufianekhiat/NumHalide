/// @file la.h
/// @brief Linear algebra operations for Halide::Func objects
///
/// Provides: matmul, dot, outer, batched operations

#pragma once

#include "common.h"
#include "shape.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Matrix Multiplication
// -----------------------------------------------------------------------------

/// @brief Matrix multiplication of two 2D Funcs
/// @param a First matrix (M x K)
/// @param shape_a Shape of first matrix
/// @param b Second matrix (K x N)
/// @param shape_b Shape of second matrix
/// @param name Function name
/// @return Result matrix (M x N)
///
/// Usage:
///   // a is 2x3, b is 3x4, result is 2x4
///   Func c = matmul(a, {2, 3}, b, {3, 4});
inline
Halide::Func matmul(Halide::Func a, const shape_t& shape_a, Halide::Func b, const shape_t& shape_b, std::string const& name = "matmul")
{
	nh_require(shape_a.rank == 2 && shape_b.rank == 2,
		"matmul requires 2D matrices, got ranks %d and %d", shape_a.rank, shape_b.rank);

	int K = shape_a.extents[1];  // cols of a = rows of b

	nh_require(shape_a.extents[1] == shape_b.extents[0],
		"matmul inner dimensions must match: (%d, %d) x (%d, %d)",
		shape_a.extents[0], shape_a.extents[1], shape_b.extents[0], shape_b.extents[1]);

	Halide::Func ret(name);
	Halide::Var x("x"), y("y");
	Halide::RDom k(0, K, "k");

	// Shape convention: [rows, cols] -> shape_a = {M, K}, shape_b = {K, N}
	// Halide convention: a(col, row) -> a(x, y) where x is column, y is row
	// result(x, y) = sum_k a(k, y) * b(x, k)

	ret(x, y) = Halide::cast(a.types()[0], 0);
	ret(x, y) += a(k, y) * b(x, k);

	return ret;
}

/// @brief Infer output shape for matmul
inline shape_t infer_matmul(const shape_t& a, const shape_t& b) {
	nh_require(a.rank == 2 && b.rank == 2,
		"matmul requires 2D matrices, got ranks %d and %d", a.rank, b.rank);
	nh_require(a.extents[1] == b.extents[0],
		"matmul inner dimensions must match: (%d, %d) x (%d, %d)",
		a.extents[0], a.extents[1], b.extents[0], b.extents[1]);

	return shape_t{ a.extents[0], b.extents[1] };
}

/// @brief Batched matrix multiplication
/// @param a First batch of matrices (batch x M x K)
/// @param shape_a Shape of first batch
/// @param b Second batch of matrices (batch x K x N)
/// @param shape_b Shape of second batch
/// @param name Function name
/// @return Result batch (batch x M x N)
///
/// Broadcasting is supported:
///   - (batch x M x K) @ (K x N) broadcasts b
///   - (M x K) @ (batch x K x N) broadcasts a
inline
Halide::Func batched_matmul(Halide::Func a, const shape_t& shape_a, Halide::Func b, const shape_t& shape_b, std::string const& name = "batched_matmul")
{
	// Handle various rank combinations
	bool a_is_batched = (shape_a.rank == 3);
	bool b_is_batched = (shape_b.rank == 3);

	nh_require(shape_a.rank >= 2 && shape_a.rank <= 3,
		"batched_matmul requires 2D or 3D input, got rank %d", shape_a.rank);
	nh_require(shape_b.rank >= 2 && shape_b.rank <= 3,
		"batched_matmul requires 2D or 3D input, got rank %d", shape_b.rank);

	int batch = 1;
	int M, K_a, K_b, N;

	if (a_is_batched) {
		batch = shape_a.extents[0];
		M = shape_a.extents[1];
		K_a = shape_a.extents[2];
	} else {
		M = shape_a.extents[0];
		K_a = shape_a.extents[1];
	}

	if (b_is_batched) {
		if (a_is_batched) {
			nh_require(shape_b.extents[0] == batch,
				"Batch dimensions must match: %d vs %d", batch, shape_b.extents[0]);
		} else {
			batch = shape_b.extents[0];
		}
		K_b = shape_b.extents[1];
		N = shape_b.extents[2];
	} else {
		K_b = shape_b.extents[0];
		N = shape_b.extents[1];
	}

	nh_require(K_a == K_b,
		"Inner dimensions must match: %d vs %d", K_a, K_b);

	int K = K_a;

	Halide::Func ret(name);
	Halide::Var x("x"), y("y"), z("z");
	Halide::RDom k(0, K, "k");

	// result(col, row, batch) = sum_k a(k, row, batch) * b(col, k, batch)
	// with broadcasting for non-batched inputs

	ret(x, y, z) = Halide::cast(a.types()[0], 0);

	if (a_is_batched && b_is_batched) {
		ret(x, y, z) += a(k, y, z) * b(x, k, z);
	} else if (a_is_batched) {
		// b is 2D, broadcast
		ret(x, y, z) += a(k, y, z) * b(x, k);
	} else if (b_is_batched) {
		// a is 2D, broadcast
		ret(x, y, z) += a(k, y) * b(x, k, z);
	} else {
		// Both 2D - shouldn't reach here, use regular matmul
		ret(x, y, z) += a(k, y) * b(x, k);
	}

	return ret;
}

/// @brief Infer output shape for batched matmul
inline shape_t infer_batched_matmul(const shape_t& a, const shape_t& b) {
	bool a_is_batched = (a.rank == 3);
	bool b_is_batched = (b.rank == 3);

	int batch = 1;
	int M, K_a, K_b, N;

	if (a_is_batched) {
		batch = a.extents[0];
		M = a.extents[1];
		K_a = a.extents[2];
	} else {
		M = a.extents[0];
		K_a = a.extents[1];
	}

	if (b_is_batched) {
		if (a_is_batched) {
			nh_require(b.extents[0] == batch, "Batch dimensions must match");
		} else {
			batch = b.extents[0];
		}
		K_b = b.extents[1];
		N = b.extents[2];
	} else {
		K_b = b.extents[0];
		N = b.extents[1];
	}

	nh_require(K_a == K_b, "Inner dimensions must match");

	return shape_t{ batch, M, N };
}

// -----------------------------------------------------------------------------
// Dot Product
// -----------------------------------------------------------------------------

/// @brief Dot product of two 1D vectors
/// @param a First vector
/// @param shape_a Shape of first vector (1D)
/// @param b Second vector
/// @param shape_b Shape of second vector (1D)
/// @param name Function name
/// @return Scalar result as 1D Func with single element
///
/// Usage:
///   Func c = dot(a, {5}, b, {5});
///   // c(0) contains the dot product
inline
Halide::Func dot(Halide::Func a, const shape_t& shape_a, Halide::Func b, const shape_t& shape_b, std::string const& name = "dot")
{
	nh_require(shape_a.rank == 1 && shape_b.rank == 1,
		"dot requires 1D vectors, got ranks %d and %d", shape_a.rank, shape_b.rank);
	nh_require(shape_a.extents[0] == shape_b.extents[0],
		"dot requires same length vectors: %d vs %d", shape_a.extents[0], shape_b.extents[0]);

	int N = shape_a.extents[0];

	Halide::Func ret(name);
	Halide::Var x("x");
	Halide::RDom i(0, N, "i");

	ret(x) = Halide::cast(a.types()[0], 0);
	ret(x) += a(i) * b(i);

	return ret;
}

/// @brief Vector-matrix dot product (1D @ 2D)
/// @param a Vector (N)
/// @param shape_a Shape of vector
/// @param b Matrix (N x M)
/// @param shape_b Shape of matrix
/// @param name Function name
/// @return Result vector (M)
inline
Halide::Func dot(Halide::Func a, const shape_t& shape_a, Halide::Func b, const shape_t& shape_b, int /*dummy*/, std::string const& name = "dot_vm")
{
	nh_require(shape_a.rank == 1 && shape_b.rank == 2,
		"dot requires 1D vector and 2D matrix, got ranks %d and %d", shape_a.rank, shape_b.rank);
	nh_require(shape_a.extents[0] == shape_b.extents[0],
		"dot: vector length must match matrix rows: %d vs %d", shape_a.extents[0], shape_b.extents[0]);

	int N = shape_a.extents[0];

	Halide::Func ret(name);
	Halide::Var x("x");
	Halide::RDom k(0, N, "k");

	// a is 1D: a(k)
	// b is 2D: b(col, row) = b(x, k)
	// result is 1D: ret(x) = sum_k a(k) * b(x, k)

	ret(x) = Halide::cast(a.types()[0], 0);
	ret(x) += a(k) * b(x, k);

	return ret;
}

// -----------------------------------------------------------------------------
// Outer Product
// -----------------------------------------------------------------------------

/// @brief Outer product of two 1D vectors
/// @param a First vector (M)
/// @param shape_a Shape of first vector
/// @param b Second vector (N)
/// @param shape_b Shape of second vector
/// @param name Function name
/// @return Matrix result (M x N)
///
/// Usage:
///   // a is (3), b is (4), result is (3, 4)
///   Func c = outer(a, {3}, b, {4});
inline
Halide::Func outer(Halide::Func a, const shape_t& shape_a, Halide::Func b, const shape_t& shape_b, std::string const& name = "outer")
{
	nh_require(shape_a.rank == 1 && shape_b.rank == 1,
		"outer requires 1D vectors, got ranks %d and %d", shape_a.rank, shape_b.rank);

	Halide::Func ret(name);
	Halide::Var x("x"), y("y");

	// result(col, row) = a(row) * b(col)
	// In shape terms: result[row, col] = a[row] * b[col]
	// Halide: result(x, y) where x=col, y=row
	ret(x, y) = a(y) * b(x);

	return ret;
}

/// @brief Infer output shape for outer product
inline shape_t infer_outer(const shape_t& a, const shape_t& b) {
	nh_require(a.rank == 1 && b.rank == 1,
		"outer requires 1D vectors, got ranks %d and %d", a.rank, b.rank);
	return shape_t{ a.extents[0], b.extents[0] };
}

// -----------------------------------------------------------------------------
// Matrix-Vector Operations
// -----------------------------------------------------------------------------

/// @brief Matrix-vector multiplication
/// @param mat Matrix (M x N)
/// @param shape_mat Shape of matrix
/// @param vec Vector (N)
/// @param shape_vec Shape of vector
/// @param name Function name
/// @return Result vector (M)
inline
Halide::Func matvec(Halide::Func mat, const shape_t& shape_mat, Halide::Func vec, const shape_t& shape_vec, std::string const& name = "matvec")
{
	nh_require(shape_mat.rank == 2 && shape_vec.rank == 1,
		"matvec requires 2D matrix and 1D vector, got ranks %d and %d", shape_mat.rank, shape_vec.rank);
	nh_require(shape_mat.extents[1] == shape_vec.extents[0],
		"matvec: matrix columns must match vector length: %d vs %d", shape_mat.extents[1], shape_vec.extents[0]);

	int N = shape_mat.extents[1];

	Halide::Func ret(name);
	Halide::Var y("y");
	Halide::RDom k(0, N, "k");

	// mat(col, row) = mat(k, y)
	// vec(k)
	// result(y) = sum_k mat(k, y) * vec(k)

	ret(y) = Halide::cast(mat.types()[0], 0);
	ret(y) += mat(k, y) * vec(k);

	return ret;
}

/// @brief Infer output shape for matvec
inline shape_t infer_matvec(const shape_t& mat, const shape_t& vec) {
	nh_require(mat.rank == 2 && vec.rank == 1,
		"matvec requires 2D matrix and 1D vector");
	nh_require(mat.extents[1] == vec.extents[0],
		"matvec: matrix columns must match vector length");
	return shape_t{ mat.extents[0] };
}

/// @brief Matrix multiply with a RUNTIME contraction length, in the
/// library's own convention: ret(x, y) += a(k, y) * b(x, k). The output
/// extents flow from the realization; only K shapes the reduction.
inline
Halide::Func matmul(Halide::Func a, Halide::Func b, Halide::Expr K, std::string const& name = "matmul_rt")
{
	Halide::Func ret(name);
	Halide::Var x("x"), y("y");
	Halide::RDom k(0, K, "k_" + name);

	ret(x, y) = Halide::cast(a.types()[0], 0);
	ret(x, y) += a(k, y) * b(x, k);

	return ret;
}

/// @brief Matrix-vector multiply with a RUNTIME length, library convention:
/// ret(y) += mat(k, y) * vec(k)
inline
Halide::Func matvec(Halide::Func mat, Halide::Func vec, Halide::Expr n, std::string const& name = "matvec_rt")
{
	Halide::Func ret(name);
	Halide::Var y("y");
	Halide::RDom k(0, n, "k_" + name);

	ret(y) = Halide::cast(mat.types()[0], 0);
	ret(y) += mat(k, y) * vec(k);

	return ret;
}

// -----------------------------------------------------------------------------
// Trace and Diagonal
// -----------------------------------------------------------------------------

/// @brief Compute the trace of a square matrix
/// @param mat Square matrix (N x N)
/// @param shape_mat Shape of matrix
/// @param name Function name
/// @return Scalar result as 1D Func with single element
inline
Halide::Func trace(Halide::Func mat, const shape_t& shape_mat, std::string const& name = "trace")
{
	nh_require(shape_mat.rank == 2,
		"trace requires 2D matrix, got rank %d", shape_mat.rank);
	nh_require(shape_mat.extents[0] == shape_mat.extents[1],
		"trace requires square matrix, got %dx%d", shape_mat.extents[0], shape_mat.extents[1]);

	int N = shape_mat.extents[0];

	Halide::Func ret(name);
	Halide::Var x("x");
	Halide::RDom i(0, N, "i");

	ret(x) = Halide::cast(mat.types()[0], 0);
	ret(x) += mat(i, i);

	return ret;
}

/// @brief Trace with a RUNTIME dimension: sum of mat(i, i), i in [0, n)
/// @param n Diagonal length as a runtime expression (e.g. a buffer extent)
inline
Halide::Func trace(Halide::Func mat, Halide::Expr n, std::string const& name = "trace")
{
	Halide::Func ret(name);
	Halide::Var x("x");
	Halide::RDom i(0, n, "i_" + name);

	ret(x) = Halide::cast(mat.types()[0], 0);
	ret(x) += mat(i, i);

	return ret;
}

/// @brief Extract the diagonal of a matrix
/// @param mat Matrix (M x N)
/// @param shape_mat Shape of matrix
/// @param name Function name
/// @return Diagonal as 1D vector (min(M, N))
inline
Halide::Func diag(Halide::Func mat, const shape_t& shape_mat, std::string const& name = "diag")
{
	nh_require(shape_mat.rank == 2,
		"diag requires 2D matrix, got rank %d", shape_mat.rank);

	Halide::Func ret(name);
	Halide::Var x("x");

	// mat(col, row) -> diagonal is mat(i, i)
	ret(x) = mat(x, x);

	return ret;
}

/// @brief Create a diagonal matrix from a vector
/// @param vec Vector (N)
/// @param shape_vec Shape of vector
/// @param name Function name
/// @return Diagonal matrix (N x N)
inline
Halide::Func diag_matrix(Halide::Func vec, const shape_t& shape_vec, std::string const& name = "diag_matrix")
{
	nh_require(shape_vec.rank == 1,
		"diag_matrix requires 1D vector, got rank %d", shape_vec.rank);

	Halide::Func ret(name);
	Halide::Var x("x"), y("y");

	ret(x, y) = Halide::select(x == y, vec(x), Halide::cast(vec.types()[0], 0));

	return ret;
}

/// @brief Infer output shape for diag
inline shape_t infer_diag(const shape_t& mat) {
	nh_require(mat.rank == 2, "diag requires 2D matrix");
	int min_dim = std::min(mat.extents[0], mat.extents[1]);
	return shape_t{ min_dim };
}

/// @brief Infer output shape for diag_matrix
inline shape_t infer_diag_matrix(const shape_t& vec) {
	nh_require(vec.rank == 1, "diag_matrix requires 1D vector");
	return shape_t{ vec.extents[0], vec.extents[0] };
}

// -----------------------------------------------------------------------------
// Norms
// -----------------------------------------------------------------------------

/// @brief Compute L2 (Euclidean) norm of a vector
/// @param vec Vector
/// @param shape_vec Shape of vector (1D)
/// @param name Function name
/// @return Scalar norm as 1D Func
inline
Halide::Func norm(Halide::Func vec, const shape_t& shape_vec, std::string const& name = "norm")
{
	nh_require(shape_vec.rank == 1,
		"vector norm requires 1D input, got rank %d", shape_vec.rank);

	int N = shape_vec.extents[0];

	Halide::Func ret(name);
	Halide::Var x("x");
	Halide::RDom i(0, N, "i");

	Halide::Func sum_sq("sum_sq");
	sum_sq(x) = Halide::cast<float>(0);
	sum_sq(x) += Halide::cast<float>(vec(i)) * Halide::cast<float>(vec(i));

	ret(x) = Halide::sqrt(sum_sq(x));

	return ret;
}

/// @brief L2 vector norm with a RUNTIME length, in the input's own type
inline
Halide::Func norm(Halide::Func vec, Halide::Expr n, std::string const& name = "norm")
{
	Halide::Type type = vec.types()[0];
	Halide::Func ret(name);
	Halide::Var x("x");
	Halide::RDom i(0, n, "i_" + name);

	Halide::Func sum_sq(name + "_sum_sq");
	sum_sq(x) = Halide::cast(type, 0);
	sum_sq(x) += vec(i) * vec(i);

	ret(x) = Halide::sqrt(sum_sq(x));

	return ret;
}

/// @brief Compute Frobenius norm of a matrix
/// @param mat Matrix
/// @param shape_mat Shape of matrix (2D)
/// @param name Function name
/// @return Scalar norm as 1D Func
inline
Halide::Func frobenius_norm(Halide::Func mat, const shape_t& shape_mat, std::string const& name = "frobenius_norm")
{
	nh_require(shape_mat.rank == 2,
		"frobenius_norm requires 2D matrix, got rank %d", shape_mat.rank);

	int M = shape_mat.extents[0];
	int N = shape_mat.extents[1];

	Halide::Func ret(name);
	Halide::Var x("x");
	Halide::RDom r(0, N, 0, M, "r");

	Halide::Func sum_sq("sum_sq");
	sum_sq(x) = Halide::cast<float>(0);
	sum_sq(x) += Halide::cast<float>(mat(r.x, r.y)) * Halide::cast<float>(mat(r.x, r.y));

	ret(x) = Halide::sqrt(sum_sq(x));

	return ret;
}

/// @brief Frobenius norm with RUNTIME extents, in the input's own type.
/// The sum is permutation-invariant, so the orientation of (m, n) only
/// needs to cover the buffer.
inline
Halide::Func frobenius_norm(Halide::Func mat, Halide::Expr m, Halide::Expr n, std::string const& name = "frobenius_norm")
{
	Halide::Type type = mat.types()[0];
	Halide::Func ret(name);
	Halide::Var x("x");
	Halide::RDom r(0, m, 0, n, "r_" + name);

	Halide::Func sum_sq(name + "_sum_sq");
	sum_sq(x) = Halide::cast(type, 0);
	sum_sq(x) += mat(r.x, r.y) * mat(r.x, r.y);

	ret(x) = Halide::sqrt(sum_sq(x));

	return ret;
}

/// @brief Compute norm along an axis
/// @param f Input Func
/// @param in_shape Shape of input
/// @param axis Axis to compute norm along
/// @param name Function name
/// @return Func with norms
inline
Halide::Func norm(Halide::Func f, const shape_t& in_shape, int axis, std::string const& name = "norm_axis")
{
	int norm_axis = normalized_axis(axis, in_shape.rank);
	int extent = in_shape.extents[norm_axis];

	Halide::RDom r(0, extent);

	Halide::Func ret(name);
	std::vector<Halide::Var> out_vars;
	int out_rank = in_shape.rank - 1;
	for (int i = 0; i < out_rank; ++i) {
		out_vars.push_back(Halide::Var());
	}

	// Build input arguments
	std::vector<Halide::Expr> in_args;
	int out_var_idx = 0;
	for (int shape_dim = in_shape.rank - 1; shape_dim >= 0; --shape_dim) {
		if (shape_dim == norm_axis) {
			in_args.push_back(r);
		} else {
			in_args.push_back(out_vars[out_rank - 1 - out_var_idx]);
			out_var_idx++;
		}
	}

	Halide::Func sum_sq("sum_sq");
	sum_sq(out_vars) = Halide::cast<float>(0);
	sum_sq(out_vars) += Halide::cast<float>(f(in_args)) * Halide::cast<float>(f(in_args));

	ret(out_vars) = Halide::sqrt(sum_sq(out_vars));

	return ret;
}

// -----------------------------------------------------------------------------
// Triangular Matrices
// -----------------------------------------------------------------------------

/// @brief Extract upper triangular part of a matrix
/// @param mat Matrix (M x N)
/// @param shape_mat Shape of matrix
/// @param k Diagonal offset (0 = main diagonal, positive = above)
/// @param name Function name
/// @return Upper triangular matrix
inline
Halide::Func triu(Halide::Func mat, const shape_t& shape_mat, int k = 0, std::string const& name = "triu")
{
	nh_require(shape_mat.rank == 2,
		"triu requires 2D matrix, got rank %d", shape_mat.rank);

	Halide::Func ret(name);
	Halide::Var x("x"), y("y");

	// For upper triangular: col >= row + k
	// In Halide: x >= y + k
	ret(x, y) = Halide::select(x >= y + k, mat(x, y), Halide::cast(mat.types()[0], 0));

	return ret;
}

/// @brief Extract lower triangular part of a matrix
/// @param mat Matrix (M x N)
/// @param shape_mat Shape of matrix
/// @param k Diagonal offset (0 = main diagonal, negative = below)
/// @param name Function name
/// @return Lower triangular matrix
inline
Halide::Func tril(Halide::Func mat, const shape_t& shape_mat, int k = 0, std::string const& name = "tril")
{
	nh_require(shape_mat.rank == 2,
		"tril requires 2D matrix, got rank %d", shape_mat.rank);

	Halide::Func ret(name);
	Halide::Var x("x"), y("y");

	// For lower triangular: col <= row + k
	// In Halide: x <= y + k
	ret(x, y) = Halide::select(x <= y + k, mat(x, y), Halide::cast(mat.types()[0], 0));

	return ret;
}

// -----------------------------------------------------------------------------
// Simple Matrix Operations
// -----------------------------------------------------------------------------

/// @brief Compute determinant of a 2x2 matrix
/// @param mat 2x2 matrix
/// @param name Function name
/// @return Scalar determinant as 1D Func
inline
Halide::Func det2x2(Halide::Func mat, std::string const& name = "det")
{
	Halide::Func ret(name);
	Halide::Var x("x");

	// det = a*d - b*c where [[a,b],[c,d]]
	// mat(col, row): a=mat(0,0), b=mat(1,0), c=mat(0,1), d=mat(1,1)
	ret(x) = mat(0, 0) * mat(1, 1) - mat(1, 0) * mat(0, 1);

	return ret;
}

/// @brief Compute inverse of a 2x2 matrix
/// @param mat 2x2 matrix
/// @param name Function name
/// @return Inverted 2x2 matrix
/// @param zero_on_singular When true, a singular matrix (det == 0) yields
///        the ZERO matrix instead of inf/NaN entries.
inline
Halide::Func inv2x2(Halide::Func mat, std::string const& name = "inv",
	bool zero_on_singular = false)
{
	Halide::Func ret(name);
	Halide::Var x("x"), y("y");

	// For [[a,b],[c,d]], inv = 1/det * [[d,-b],[-c,a]]
	Halide::Type type = mat.types()[0];
	Halide::Expr det = mat(0, 0) * mat(1, 1) - mat(1, 0) * mat(0, 1);
	Halide::Expr inv_det = Halide::cast(type, 1) / det;
	if (zero_on_singular)
		inv_det = Halide::select(det != Halide::cast(type, 0),
			inv_det, Halide::cast(type, 0));

	ret(x, y) = Halide::select(
		x == 0 && y == 0, inv_det * mat(1, 1),
		Halide::select(
			x == 1 && y == 0, -inv_det * mat(1, 0),
			Halide::select(
				x == 0 && y == 1, -inv_det * mat(0, 1),
				inv_det * mat(0, 0)
			)
		)
	);

	return ret;
}

/// @brief Compute determinant of a 3x3 matrix
/// @param mat 3x3 matrix
/// @param name Function name
/// @return Scalar determinant as 1D Func
inline
Halide::Func det3x3(Halide::Func mat, std::string const& name = "det3x3")
{
	Halide::Func ret(name);
	Halide::Var x("x");

	// Using cofactor expansion along first row
	Halide::Expr a = mat(0, 0), b = mat(1, 0), c = mat(2, 0);
	Halide::Expr d = mat(0, 1), e = mat(1, 1), f = mat(2, 1);
	Halide::Expr g = mat(0, 2), h = mat(1, 2), i = mat(2, 2);

	ret(x) = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);

	return ret;
}

// -----------------------------------------------------------------------------
// Inner Product, Kronecker Product, Matrix Power
// -----------------------------------------------------------------------------

/// @brief Inner product of two 1D arrays: sum(a[i] * b[i])
/// @param a First 1D Func
/// @param b Second 1D Func
/// @param n Size of arrays
/// @param name Function name
/// @return Scalar Func (1D, size 1), index 0 = result
inline
Halide::Func inner_1d(Halide::Func a, Halide::Func b, Halide::Expr n,
    std::string const& name = "inner")
{
    Halide::Func ret(name);
    Halide::Var i("i");
    Halide::RDom r(0, n, "r_inner");

    ret(i) = Halide::cast(a.types()[0], 0);
    ret(i) += a(r) * b(r);

    return ret;
}

/// @brief Kronecker product of two 2D matrices
/// @param a Matrix A (shape_a = {Ma, Na})
/// @param shape_a Shape of A
/// @param b Matrix B (shape_b = {Mb, Nb})
/// @param shape_b Shape of B
/// @param name Function name
/// @return Matrix of shape {Ma*Mb, Na*Nb}
///
/// kron[i*Mb + m, j*Nb + n] = A[i,j] * B[m,n]
/// In Halide: ret(col, row) where col = j*Nb+n, row = i*Mb+m
///   ret(col, row) = a(col/Nb, row/Mb) * b(col%Nb, row%Mb)
inline
Halide::Func kron(Halide::Func a, const shape_t& shape_a,
    Halide::Func b, const shape_t& shape_b,
    std::string const& name = "kron")
{
    nh_require(shape_a.rank == 2 && shape_b.rank == 2,
        "kron requires 2D matrices");

    int Mb = shape_b.extents[0], Nb = shape_b.extents[1];

    Halide::Func ret(name);
    Halide::Var col("col"), row("row");

    // In Halide: a(x, y) = a[row=y, col=x], shape {Ma, Na} → a(col_a, row_a)
    // ret col = j*Nb + n, ret row = i*Mb + m
    // a col = j = col / Nb, a row = i = row / Mb
    // b col = n = col % Nb, b row = m = row % Mb

    ret(col, row) = a(col / Nb, row / Mb) * b(col % Nb, row % Mb);

    return ret;
}

/// @brief Kronecker product with RUNTIME b extents, buffer-axis form:
/// ret(x, y) = a(x / b_d0, y / b_d1) * b(x % b_d0, y % b_d1)
/// (b_d0/b_d1 are b's extents along dims 0/1; output extents flow from
/// the realization.)
inline
Halide::Func kron(Halide::Func a, Halide::Func b, Halide::Expr b_d0, Halide::Expr b_d1,
    std::string const& name = "kron_rt")
{
    Halide::Func ret(name);
    Halide::Var x("x"), y("y");
    ret(x, y) = a(x / b_d0, y / b_d1) * b(x % b_d0, y % b_d1);
    return ret;
}

/// @brief Output shape for kron
inline shape_t infer_kron(const shape_t& a, const shape_t& b) {
    nh_require(a.rank == 2 && b.rank == 2, "kron requires 2D matrices");
    return shape_t{a.extents[0] * b.extents[0], a.extents[1] * b.extents[1]};
}

/// @brief Raise a square matrix to an integer power
/// @param a Square matrix (shape = {N, N})
/// @param shape_a Shape of matrix
/// @param p Non-negative integer power
/// @param name Function name
/// @return Matrix a^p (shape = {N, N})
///
/// Uses repeated squaring for efficiency.
/// p=0 returns identity, p=1 returns a, p=2 returns a@a, etc.
inline
Halide::Func matrix_power(Halide::Func a, const shape_t& shape_a, int p,
    std::string const& name = "matrix_power")
{
    nh_require(shape_a.rank == 2 && shape_a.extents[0] == shape_a.extents[1],
        "matrix_power requires square matrix");
    nh_require(p >= 0, "matrix_power requires non-negative exponent");

    Halide::Type type = a.types()[0];

    if (p == 0) {
        // Return identity matrix
        Halide::Func ret(name);
        Halide::Var x("x"), y("y");
        ret(x, y) = Halide::select(x == y, Halide::cast(type, 1), Halide::cast(type, 0));
        return ret;
    }

    // Repeated multiplication: result = a^p using squaring
    // For simplicity, use sequential multiplication (p multiplications)
    // Could optimize with binary exponentiation for large p
    Halide::Func result = a;
    shape_t sq = shape_a;
    for (int i = 1; i < p; ++i) {
        result = matmul(result, sq, a, shape_a, name + "_mp" + std::to_string(i));
        result.compute_root();
    }

    Halide::Func ret(name);
    Halide::Var x("x"), y("y");
    ret(x, y) = result(x, y);
    return ret;
}

// -----------------------------------------------------------------------------
// Cross Product
// -----------------------------------------------------------------------------

/// @brief 3D cross product of two 3-element 1D Funcs
/// @return 3-element Func: c = a × b
inline
Halide::Func cross_3d(Halide::Func a, Halide::Func b,
    std::string const& name = "cross")
{
    Halide::Func ret(name);
    Halide::Var i("i");
    ret(i) = Halide::select(i == 0, a(1) * b(2) - a(2) * b(1),
             Halide::select(i == 1, a(2) * b(0) - a(0) * b(2),
                                    a(0) * b(1) - a(1) * b(0)));
    return ret;
}

// -----------------------------------------------------------------------------
// Tensor Dot Product
// -----------------------------------------------------------------------------

/// @brief Generalized tensor contraction
/// @param axes 0 = outer product, 1 = contract last dim of a with first dim of b
///
/// axes=0: outer product, output rank = rank_a + rank_b
/// axes=1: like matmul — contract innermost shape dim of a with outermost of b
inline
Halide::Func tensordot(Halide::Func a, const shape_t& shape_a,
    Halide::Func b, const shape_t& shape_b,
    int axes = 1,
    std::string const& name = "tensordot")
{
    int ra = shape_a.rank, rb = shape_b.rank;
    Halide::Type type = a.types()[0];

    if (axes == 0) {
        // Outer product: output Halide dims = [b dims (inner), a dims (outer)]
        int out_rank = ra + rb;
        std::vector<Halide::Var> out_vars;
        for (int i = 0; i < out_rank; ++i) out_vars.push_back(Halide::Var());
        // b uses out_vars[0..rb-1], a uses out_vars[rb..out_rank-1]
        std::vector<Halide::Expr> b_args(out_vars.begin(), out_vars.begin() + rb);
        std::vector<Halide::Expr> a_args(out_vars.begin() + rb, out_vars.end());
        Halide::Func ret(name);
        ret(out_vars) = a(a_args) * b(b_args);
        return ret;
    }

    // axes=1: contract innermost shape dim of a with outermost shape dim of b
    nh_require(axes == 1, "tensordot: only axes=0 or axes=1 supported");
    int K = shape_a.extents[ra - 1]; // = shape_b.extents[0]
    int out_rank = ra + rb - 2;

    // out_rank == 0 means dot product of two 1D arrays
    if (out_rank == 0) {
        Halide::Func ret(name);
        ret() = Halide::Internal::make_const(type, 0);
        Halide::RDom rk(0, K, "rk_td");
        ret() += a(Halide::Expr(rk)) * b(Halide::Expr(rk));
        return ret;
    }

    std::vector<Halide::Var> out_vars;
    for (int i = 0; i < out_rank; ++i) out_vars.push_back(Halide::Var());
    // a_args: [rk, out_vars[rb-1 .. out_rank-1]] (rk=a's innermost, others=a's outer dims)
    // b_args: [out_vars[0 .. rb-2], rk]          (others=b's inner dims, rk=b's outermost)
    std::vector<Halide::Expr> a_args;
    a_args.push_back(Halide::Expr(Halide::RDom(0, K, "rk_td_a")));
    for (int d = rb - 1; d < out_rank; ++d) a_args.push_back(Halide::Expr(out_vars[d]));
    std::vector<Halide::Expr> b_args;
    for (int d = 0; d < rb - 1; ++d) b_args.push_back(Halide::Expr(out_vars[d]));

    // Use a single RDom for both
    Halide::Func ret(name);
    ret(out_vars) = Halide::Internal::make_const(type, 0);
    Halide::RDom rk(0, K, "rk_td");
    std::vector<Halide::Expr> a_args2, b_args2;
    a_args2.push_back(Halide::Expr(rk));
    for (int d = rb - 1; d < out_rank; ++d) a_args2.push_back(Halide::Expr(out_vars[d]));
    for (int d = 0; d < rb - 1; ++d) b_args2.push_back(Halide::Expr(out_vars[d]));
    b_args2.push_back(Halide::Expr(rk));
    ret(out_vars) += a(a_args2) * b(b_args2);
    return ret;
}

// -----------------------------------------------------------------------------
// Normalize
// -----------------------------------------------------------------------------

/// @brief L2-normalize f along the given axis (each slice along axis becomes unit length)
/// @param f    Input Func
/// @param shape Shape of f
/// @param axis  Axis along which to normalize (default -1 = last)
/// @param eps   Small value to avoid division by zero
inline
Halide::Func normalize(Halide::Func f, const shape_t& shape, int axis = -1,
    float eps = 1e-8f, std::string const& name = "normalize")
{
    int rank = shape.rank;
    int norm_axis = (axis < 0) ? (rank + axis) : axis;
    nh_require(norm_axis >= 0 && norm_axis < rank, "normalize: axis out of range");

    int halide_norm_dim = rank - 1 - norm_axis; // Halide dim for shape axis norm_axis
    int K = shape.extents[norm_axis];           // size along the normalized axis

    std::vector<Halide::Var> vars;
    for (int i = 0; i < rank; ++i) vars.push_back(Halide::Var());

    // out_vars = vars without halide_norm_dim (the outer free variables)
    std::vector<Halide::Var> out_vars;
    for (int d = 0; d < rank; ++d) {
        if (d != halide_norm_dim) out_vars.push_back(vars[d]);
    }

    // Reduction var over the normalized dimension
    Halide::RDom rk(0, K, "rk_norm");

    // f_args for the reduction: insert rk at halide_norm_dim
    std::vector<Halide::Expr> f_args_rk;
    for (int d = 0; d < rank; ++d) {
        if (d == halide_norm_dim) f_args_rk.push_back(Halide::Expr(rk));
        else f_args_rk.push_back(Halide::Expr(vars[d]));
    }

    // Sum of squares
    Halide::Func ss(name + "_ss");
    if (out_vars.empty()) {
        ss() = 0.0f;
        ss() += f(f_args_rk) * f(f_args_rk);
    } else {
        ss(out_vars) = 0.0f;
        ss(out_vars) += f(f_args_rk) * f(f_args_rk);
    }
    ss.compute_root();

    // ss_args: index ss with the free vars (same as out_vars but as Exprs)
    std::vector<Halide::Expr> ss_args;
    for (int d = 0; d < rank; ++d) {
        if (d != halide_norm_dim) ss_args.push_back(Halide::Expr(vars[d]));
    }

    Halide::Func ret(name);
    Halide::Expr l2;
    if (ss_args.empty()) {
        l2 = Halide::sqrt(ss());
    } else {
        l2 = Halide::sqrt(ss(ss_args));
    }
    ret(vars) = f(vars) / Halide::max(l2, eps);
    return ret;
}

// -----------------------------------------------------------------------------
// 3×3 Inverse
// -----------------------------------------------------------------------------

/// @brief Compute inverse of a 3×3 matrix via cofactor / adjugate method
/// @param mat 3×3 matrix Func (mat(col, row))
/// @param name Func name
/// @return Inverted 3×3 matrix
///
/// inv = adjugate(mat) / det(mat)
/// adjugate[i,j] = cofactor[j,i]  (transpose of cofactor matrix)
///
/// mat(x,y) layout: a=mat(0,0) b=mat(1,0) c=mat(2,0)
///                  d=mat(0,1) e=mat(1,1) f=mat(2,1)
///                  g=mat(0,2) h=mat(1,2) i=mat(2,2)
/// @param zero_on_singular When true, a singular matrix (det == 0) yields
///        the ZERO matrix instead of inf/NaN entries.
inline
Halide::Func inv3x3(Halide::Func mat, std::string const& name = "inv3x3",
    bool zero_on_singular = false)
{
    Halide::Var x("x"), y("y");

    Halide::Expr a = mat(0,0), b = mat(1,0), c = mat(2,0);
    Halide::Expr d = mat(0,1), e = mat(1,1), f = mat(2,1);
    Halide::Expr g = mat(0,2), h = mat(1,2), ii = mat(2,2);

    Halide::Type type = mat.types()[0];
    Halide::Expr det = a*(e*ii - f*h) - b*(d*ii - f*g) + c*(d*h - e*g);
    Halide::Expr inv_det = Halide::cast(type, 1) / det;
    if (zero_on_singular)
        inv_det = Halide::select(det != Halide::cast(type, 0),
            inv_det, Halide::cast(type, 0));

    // inv(col=x, row=y) = adj[y,x] / det = cofactor[x,y] / det
    // Each entry: cofactor C[r,c] = (-1)^(r+c) * minor_det
    Halide::Func ret(name);
    ret(x, y) = Halide::select(
        x == 0 && y == 0, inv_det * (e*ii - f*h),
        Halide::select(x == 1 && y == 0, inv_det * (c*h  - b*ii),
        Halide::select(x == 2 && y == 0, inv_det * (b*f  - c*e),
        Halide::select(x == 0 && y == 1, inv_det * (f*g  - d*ii),
        Halide::select(x == 1 && y == 1, inv_det * (a*ii - c*g),
        Halide::select(x == 2 && y == 1, inv_det * (c*d  - a*f),
        Halide::select(x == 0 && y == 2, inv_det * (d*h  - e*g),
        Halide::select(x == 1 && y == 2, inv_det * (b*g  - a*h),
                                         inv_det * (a*e  - b*d)))))))));  // x==2, y==2
    return ret;
}

// -----------------------------------------------------------------------------
// 2×2 SVD (analytical)
// -----------------------------------------------------------------------------

/// @brief Result of a 2×2 SVD
struct SVD2x2Result {
    Halide::Func U;   ///< 2×2 left singular vectors
    Halide::Func S;   ///< 1D, two singular values (sigma0 ≥ sigma1 in general)
    Halide::Func Vt;  ///< 2×2 right singular vectors transposed
};

/// @brief Analytic SVD of a 2×2 matrix: M = U * diag(S) * Vt
/// @param mat 2×2 Func (mat(col, row))
/// @param name Base name for result Funcs
/// @return SVD2x2Result with U, S, Vt Funcs
///
/// Uses the Jacobi one-sided approach:
///   theta = 0.5 * atan2(2*(a*b + c*d), a²+c² − b²−d²)
///   V = [[cos,−sin],[sin,cos]];  sigma_k = ||M·V[:,k]||;  U = M·V·diag(1/sigma)
inline
SVD2x2Result svd2x2(Halide::Func mat, std::string const& name = "svd2x2")
{
    // mat(x,y): a=mat(0,0) b=mat(1,0) c=mat(0,1) d=mat(1,1)
    Halide::Expr a = mat(0,0), b = mat(1,0), c = mat(0,1), d = mat(1,1);

    Halide::Expr theta = 0.5f * Halide::atan2(
        2.0f * (a*b + c*d),
        a*a + c*c - b*b - d*d);
    Halide::Expr cs = Halide::cos(theta);
    Halide::Expr sn = Halide::sin(theta);

    // M * V columns:  MV[:,0] = [a·cs+b·sn, c·cs+d·sn]
    //                 MV[:,1] = [−a·sn+b·cs, −c·sn+d·cs]
    Halide::Expr mv00 = a*cs + b*sn,  mv10 = c*cs + d*sn;
    Halide::Expr mv01 = -a*sn + b*cs, mv11 = -c*sn + d*cs;

    Halide::Expr sig0 = Halide::sqrt(mv00*mv00 + mv10*mv10);
    Halide::Expr sig1 = Halide::sqrt(mv01*mv01 + mv11*mv11);

    Halide::Var x("x"), y("y"), k("k");

    Halide::Func S(name + "_S");
    S(k) = Halide::select(k == 0, sig0, sig1);

    // U = MV * diag(1/sig)
    Halide::Func U(name + "_U");
    U(x, y) = Halide::select(
        x == 0 && y == 0, mv00 / sig0,
        Halide::select(x == 0 && y == 1, mv10 / sig0,
        Halide::select(x == 1 && y == 0, mv01 / sig1,
                                          mv11 / sig1)));  // x==1, y==1

    // Vt = V^T:  V=[[cs,−sn],[sn,cs]] → Vt=[[cs,sn],[−sn,cs]]
    Halide::Func Vt(name + "_Vt");
    Vt(x, y) = Halide::select(
        x == 0 && y == 0,  cs,
        Halide::select(x == 1 && y == 0,  sn,
        Halide::select(x == 0 && y == 1, -sn,
                                          cs)));  // x==1, y==1

    return {U, S, Vt};
}

// -----------------------------------------------------------------------------
// Cholesky decomposition
// -----------------------------------------------------------------------------

/// @brief Cholesky decomposition: returns lower-triangular L such that A = L·Lᵀ
/// @param A    Positive-definite symmetric matrix Func (A(col, row))
/// @param n    Matrix size (n×n)
/// @param name Base Func name
/// @param eps  When > 0, every sqrt argument is floored: sqrt(max(x, eps)).
///             The diagonal is then ≥ sqrt(eps) > 0, so the column divisions
///             by L[j,j] are guarded too. 0 keeps the unguarded behavior.
/// @return Lower-triangular Func L(col, row) with L[i≥j,j] ≠ 0, L[i<j,j] = 0
///
/// Column-by-column sequential algorithm using compute_root staging.
/// Works for any n; internally creates O(n) intermediate Funcs.
/// Computes in A's element type (f32, f64, ...).
inline
Halide::Func cholesky(Halide::Func A, int n, std::string const& name = "cholesky",
    float eps = 0.0f)
{
    Halide::Var col("col"), row("row");

    Halide::Type type = A.types()[0];
    Halide::Expr zero = Halide::cast(type, 0);
    Halide::Expr eps_e = Halide::cast(type, Halide::Expr(static_cast<double>(eps)));

    // Initial L is all zeros
    Halide::Func L_cur(name + "_init");
    L_cur(col, row) = zero;
    L_cur.compute_root();

    for (int j = 0; j < n; ++j) {
        std::string sj = std::to_string(j);

        // -- Diagonal element L[j,j] = sqrt(A[j,j] - sum_{k<j} L[j,k]^2) --
        Halide::Func ss(name + "_ss" + sj);
        ss() = zero;
        if (j > 0) {
            Halide::RDom rk(0, j, "rkss" + sj);
            // L_cur(col=rk, row=j) = L_{row=j, col=k}
            ss() += L_cur(rk, j) * L_cur(rk, j);
        }
        ss.compute_root();

        Halide::Func diag_j(name + "_djj" + sj);
        if (eps > 0.0f)
            diag_j() = Halide::sqrt(Halide::max(A(j, j) - ss(), eps_e));
        else
            diag_j() = Halide::sqrt(A(j, j) - ss());
        diag_j.compute_root();

        // -- Off-diagonal column j: dots(i) = sum_{k<j} L[i,k]*L[j,k] --
        Halide::Func dots(name + "_dots" + sj);
        dots(row) = zero;
        if (j > 0) {
            Halide::RDom rk(0, j, "rkd" + sj);
            dots(row) += L_cur(rk, row) * L_cur(rk, j);
        }
        dots.compute_root();

        // -- Build updated L matrix --
        Halide::Func L_next(name + "_step" + sj);
        L_next(col, row) = L_cur(col, row);   // copy previous
        L_next(j, j)     = diag_j();          // diagonal
        if (j < n - 1) {
            Halide::RDom ri(j + 1, n - j - 1, "ri_chol" + sj);
            // L[row=ri, col=j] = (A[ri,j] - dots(ri)) / L[j,j]
            L_next(j, ri) = (A(j, ri) - dots(ri)) / diag_j();
        }
        L_next.compute_root();
        L_cur = L_next;
    }

    Halide::Func ret(name);
    ret(col, row) = L_cur(col, row);
    return ret;
}

// -----------------------------------------------------------------------------
// QR decomposition via modified Gram-Schmidt
// -----------------------------------------------------------------------------

/// @brief Result of a QR decomposition
struct QRResult {
    Halide::Func Q;  ///< m×n orthonormal matrix
    Halide::Func R;  ///< n×n upper-triangular matrix
};

/// @brief QR decomposition via modified Gram-Schmidt
/// @param A    m×n matrix Func (A(col, row), m ≥ n)
/// @param m    Number of rows
/// @param n    Number of columns (≤ m)
/// @param name Base Func name
/// @param eps  When > 0, the column-norm sqrt is floored: sqrt(max(ss, eps)).
///             The Q normalization then divides by that guarded norm
///             (≥ sqrt(eps) > 0). 0 keeps the unguarded behavior.
/// @return QRResult {Q (m×n), R (n×n)} such that Q·R = A and Qᵀ·Q = I
///
/// Uses modified Gram-Schmidt (numerically more stable than classical GS).
/// Internally creates O(n) compute_root stages.
/// Computes in A's element type (f32, f64, ...).
inline
QRResult qr_gs(Halide::Func A, int m, int n, std::string const& name = "qr",
    float eps = 0.0f)
{
    nh_require(m >= n, "qr_gs: requires m >= n, got m=%d n=%d", m, n);

    Halide::Var col("col"), row("row");

    Halide::Type type = A.types()[0];
    Halide::Expr zero = Halide::cast(type, 0);
    Halide::Expr eps_e = Halide::cast(type, Halide::Expr(static_cast<double>(eps)));

    // Working copy of A (gets modified column by column)
    Halide::Func Aw(name + "_aw0");
    Aw(col, row) = A(col, row);
    Aw.compute_root();

    // Q and R initialised to zero
    Halide::Func Q(name + "_q0");
    Q(col, row) = zero;
    Q.compute_root();

    Halide::Func R(name + "_r0");
    R(col, row) = zero;
    R.compute_root();

    for (int k = 0; k < n; ++k) {
        std::string sk = std::to_string(k);

        // 1. Norm of column k of Aw
        Halide::RDom r_norm(0, m, "rnorm" + sk);
        Halide::Func ss_k(name + "_ssk" + sk);
        ss_k() = zero;
        ss_k() += Aw(k, r_norm) * Aw(k, r_norm);
        ss_k.compute_root();

        Halide::Func nk(name + "_nk" + sk);
        if (eps > 0.0f)
            nk() = Halide::sqrt(Halide::max(ss_k(), eps_e));
        else
            nk() = Halide::sqrt(ss_k());
        nk.compute_root();

        // 2. Q[:,k] = Aw[:,k] / nk  and  R[k,k] = nk
        {
            Halide::Func Q_new(name + "_q" + sk);
            Q_new(col, row) = Q(col, row);
            {
                Halide::RDom rr(0, m, "rq" + sk);
                Q_new(k, rr) = Aw(k, rr) / nk();
            }
            Q_new.compute_root();
            Q = Q_new;
        }
        {
            Halide::Func R_new(name + "_rd" + sk);
            R_new(col, row) = R(col, row);
            R_new(k, k) = nk();   // R[row=k, col=k]
            R_new.compute_root();
            R = R_new;
        }

        if (k < n - 1) {
            // 3. Dot products: dp[j] = Q[:,k]·Aw[:,j]  for j > k
            Halide::RDom r_dp(0, m, "rdp" + sk);
            Halide::Func dp(name + "_dp" + sk);
            dp(col) = zero;
            dp(col) += Q(k, r_dp) * Aw(col, r_dp);
            dp.compute_root();

            // 4. R[k,j] = dp[j] for j in [k+1, n)
            {
                Halide::Func R_new(name + "_ru" + sk);
                R_new(col, row) = R(col, row);
                {
                    Halide::RDom rj(k + 1, n - k - 1, "rru" + sk);
                    R_new(rj, k) = dp(rj);  // R(col=j, row=k) = R[k,j]
                }
                R_new.compute_root();
                R = R_new;
            }

            // 5. Aw[:,j] -= dp[j] * Q[:,k]  for j in [k+1, n)
            {
                Halide::Func Aw_new(name + "_aw" + sk);
                Aw_new(col, row) = Aw(col, row);
                {
                    Halide::RDom r_up(k + 1, n - k - 1, 0, m, "rawu" + sk);
                    Aw_new(r_up.x, r_up.y) =
                        Aw(r_up.x, r_up.y) - dp(r_up.x) * Q(k, r_up.y);
                }
                Aw_new.compute_root();
                Aw = Aw_new;
            }
        }
    }

    Halide::Func Q_out(name + "_Q");
    Q_out(col, row) = Q(col, row);

    Halide::Func R_out(name + "_R");
    R_out(col, row) = R(col, row);

    return {Q_out, R_out};
}

// -----------------------------------------------------------------------------
// Triangular solvers
// -----------------------------------------------------------------------------

/// @brief Back-substitution: solve upper-triangular R·x = y
/// @param R  n×n upper-triangular Func (R(col, row))
/// @param y  1D n-vector Func
/// @param n  System size
/// @param eps When > 0, a diagonal pivot with |R[k,k]| ≤ eps is replaced by 1
///            for the division (finite fallback instead of inf/NaN).
///            0 keeps the unguarded behavior.
/// @return   1D solution Func x(i)
///
/// Computes in R's element type (f32, f64, ...).
inline Halide::Func back_sub(Halide::Func R, Halide::Func y, int n,
    std::string const& name = "back_sub", float eps = 0.0f)
{
    Halide::Var i("i");

    Halide::Type type = R.types()[0];
    Halide::Expr zero = Halide::cast(type, 0);
    Halide::Expr eps_e = Halide::cast(type, Halide::Expr(static_cast<double>(eps)));

    Halide::Func x_cur(name + "_x0");
    x_cur(i) = zero;
    x_cur.compute_root();

    for (int k = n - 1; k >= 0; --k) {
        std::string sk = std::to_string(k);

        // dot = Σ_{j>k} R[k,j] * x[j]  →  R(col=j, row=k) = R(j, k)
        Halide::Func dot(name + "_dot" + sk);
        dot() = zero;
        if (k < n - 1) {
            Halide::RDom rj(k + 1, n - k - 1, "rjbs" + sk);
            dot() += R(rj, k) * x_cur(rj);
        }
        dot.compute_root();

        Halide::Expr rkk = R(k, k);
        if (eps > 0.0f)
            rkk = Halide::select(Halide::abs(rkk) > eps_e, rkk,
                Halide::cast(type, 1));

        Halide::Func x_next(name + "_xk" + sk);
        x_next(i)  = x_cur(i);
        x_next(k)  = (y(k) - dot()) / rkk;
        x_next.compute_root();
        x_cur = x_next;
    }
    return x_cur;
}

/// @brief Forward-substitution: solve lower-triangular L·y = b
/// @param L  n×n lower-triangular Func (L(col, row))
/// @param b  1D n-vector Func
/// @param n  System size
/// @return   1D solution Func y(i)
/// @param eps When > 0, a diagonal pivot with |L[k,k]| <= eps is replaced by 1
///            for the division (finite fallback instead of inf/NaN).
///            0 keeps the unguarded behavior.
///
/// Computes in L's element type (f32, f64, ...).
inline Halide::Func fwd_sub(Halide::Func L, Halide::Func b, int n,
    std::string const& name = "fwd_sub", float eps = 0.0f)
{
    Halide::Var i("i");

    Halide::Type type = L.types()[0];
    Halide::Expr zero = Halide::cast(type, 0);
    Halide::Expr eps_e = Halide::cast(type, Halide::Expr(static_cast<double>(eps)));

    Halide::Func y_cur(name + "_y0");
    y_cur(i) = zero;
    y_cur.compute_root();

    for (int k = 0; k < n; ++k) {
        std::string sk = std::to_string(k);

        Halide::Func dot(name + "_dot" + sk);
        dot() = zero;
        if (k > 0) {
            Halide::RDom rj(0, k, "rjfs" + sk);
            dot() += L(rj, k) * y_cur(rj);
        }
        dot.compute_root();

        Halide::Expr lkk = L(k, k);
        if (eps > 0.0f)
            lkk = Halide::select(Halide::abs(lkk) > eps_e, lkk,
                Halide::cast(type, 1));

        Halide::Func y_next(name + "_yk" + sk);
        y_next(i) = y_cur(i);
        y_next(k) = (b(k) - dot()) / lkk;
        y_next.compute_root();
        y_cur = y_next;
    }
    return y_cur;
}

// -----------------------------------------------------------------------------
// Full SVD via one-sided Jacobi iteration
// -----------------------------------------------------------------------------

/// @brief Result of a full SVD: M = U * diag(S) * Vt
struct SVDResult {
    Halide::Func U;   ///< m×n left singular vectors
    Halide::Func S;   ///< 1D, n singular values
    Halide::Func Vt;  ///< n×n right singular vectors transposed
};

/// @brief Full SVD of an m×n matrix via one-sided Jacobi sweeps
/// @param A        m×n matrix Func (A(col, row), m ≥ n)
/// @param m        Number of rows
/// @param n        Number of columns
/// @param n_sweeps Number of Jacobi sweeps (more = more accurate, default 10)
/// @param name     Base name
/// @return SVDResult {U (m×n), S (1D size n), Vt (n×n)}
///
/// Each sweep applies n*(n-1)/2 Jacobi column-rotation pairs.
/// Convergence: off-diagonal entries of W^T W → 0.
/// Computes in A's element type (f32, f64, ...).
/// Note: internally creates O(n_sweeps * n²) compute_root stages; keep n ≤ 8.
inline SVDResult svd_jacobi(Halide::Func A, int m, int n,
    int n_sweeps = 10, std::string const& name = "svd")
{
    Halide::Var col("col"), row("row"), k("k");

    Halide::Type type = A.types()[0];
    Halide::Expr zero = Halide::Internal::make_const(type, 0.0);
    Halide::Expr one  = Halide::Internal::make_const(type, 1.0);

    // Working matrix W (m×n): accumulates column rotations
    Halide::Func W(name + "_W0");
    W(col, row) = Halide::cast(type, A(col, row));
    W.compute_root();

    // Right singular vector matrix V (n×n): starts as identity
    Halide::Func V(name + "_V0");
    V(col, row) = Halide::select(col == row, one, zero);
    V.compute_root();

    for (int sweep = 0; sweep < n_sweeps; ++sweep) {
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const std::string tag = name
                    + "_s" + std::to_string(sweep)
                    + "p" + std::to_string(p)
                    + "q" + std::to_string(q);

                // Column dot products
                Halide::Func alpha(tag + "_a");
                alpha() = zero;
                { Halide::RDom ra(0, m, "ra_" + tag); alpha() += W(p, ra) * W(p, ra); }
                alpha.compute_root();

                Halide::Func beta(tag + "_b");
                beta() = zero;
                { Halide::RDom rb(0, m, "rb_" + tag); beta()  += W(q, rb) * W(q, rb); }
                beta.compute_root();

                Halide::Func gam(tag + "_g");
                gam() = zero;
                { Halide::RDom rg(0, m, "rg_" + tag); gam()   += W(p, rg) * W(q, rg); }
                gam.compute_root();

                // Jacobi rotation: ζ=(β−α)/(2γ), t=sgn(ζ)/(|ζ|+√(1+ζ²))
                // cs = 1/√(1+t²), sn = cs·t   (skip when γ≈0)
                Halide::Expr g    = gam();
                Halide::Expr skip = Halide::abs(g) < Halide::Internal::make_const(type, 1e-10);
                Halide::Expr zeta = (beta() - alpha()) / (Halide::Internal::make_const(type, 2.0) * g);
                Halide::Expr abz  = Halide::abs(zeta);
                Halide::Expr t    = Halide::select(zeta >= zero, one, Halide::Internal::make_const(type, -1.0))
                                    / (abz + Halide::sqrt(one + zeta * zeta));

                Halide::Func cs_f(tag + "_cs");
                cs_f() = Halide::select(skip, one,
                    one / Halide::sqrt(one + t * t));
                cs_f.compute_root();

                Halide::Func sn_f(tag + "_sn");
                sn_f() = Halide::select(skip, zero, cs_f() * t);
                sn_f.compute_root();

                // Update W[:,p] and W[:,q]  (using old W on RHS).
                // SIGN CONVENTION: t above solves t^2 + 2*zeta*t - 1 = 0,
                // which annihilates the p.q dot product ONLY with the
                // standard update wp' = c*wp - s*wq, wq' = s*wp + c*wq.
                // The previous opposite-sign update left gamma unreduced
                // and the sweep OSCILLATED instead of converging
                // (measured: U^T*U off-diagonal ~0.39 regardless of
                // sweeps).
                Halide::Func W_next(tag + "_W");
                W_next(col, row) = W(col, row);
                { Halide::RDom rp(0, m, "rWp_" + tag);
                  W_next(p, rp) = cs_f() * W(p, rp) - sn_f() * W(q, rp); }
                { Halide::RDom rq(0, m, "rWq_" + tag);
                  W_next(q, rq) = sn_f() * W(p, rq) + cs_f() * W(q, rq); }
                W_next.compute_root();
                W = W_next;

                // Update V[:,p] and V[:,q] with the SAME rotation
                Halide::Func V_next(tag + "_V");
                V_next(col, row) = V(col, row);
                { Halide::RDom rvp(0, n, "rVp_" + tag);
                  V_next(p, rvp) = cs_f() * V(p, rvp) - sn_f() * V(q, rvp); }
                { Halide::RDom rvq(0, n, "rVq_" + tag);
                  V_next(q, rvq) = sn_f() * V(p, rvq) + cs_f() * V(q, rvq); }
                V_next.compute_root();
                V = V_next;
            }
        }
    }

    // Singular values: S[k] = ||W[:,k]||
    Halide::Func ss_sv(name + "_ss");
    { Halide::RDom r_sv(0, m, "rsv"); ss_sv(k) = zero; ss_sv(k) += W(k, r_sv) * W(k, r_sv); }
    ss_sv.compute_root();

    Halide::Func S_raw(name + "_Sraw");
    S_raw(k) = Halide::sqrt(ss_sv(k));
    S_raw.compute_root();

    // DESCENDING order (numpy convention): stable rank of each column,
    // then gather S / U columns / V columns through the permutation.
    Halide::Func rank_f(name + "_rank");
    rank_f(k) = Halide::cast<int32_t>(0);
    { Halide::RDom rr(0, n, "rrank_" + name);
      rank_f(k) += Halide::select(
          S_raw(rr) > S_raw(k) || (S_raw(rr) == S_raw(k) && rr < k),
          Halide::cast<int32_t>(1), Halide::cast<int32_t>(0)); }
    rank_f.compute_root();

    // Gathers use MULTIPLIED 0/1 indicators, not select: reverse-mode AD
    // synthesizes a float32 zero for the inactive select branch, which
    // type-clashes with the f64 derivative of a division inside the arm.
    // The indicator form derives cleanly in every type; the divisor is
    // floored so an exactly-singular input stays finite.
    Halide::Expr tiny = Halide::Internal::make_const(type, 1e-30);

    Halide::Func S(name + "_S");
    { Halide::RDom rs(0, n, "rgs_" + name);
      S(k) = zero;
      S(k) += S_raw(rs) * Halide::cast(type, rank_f(rs) == k); }
    S.compute_root();

    // U[:,k] = W[:,perm(k)] / S[perm(k)]
    Halide::Func U(name + "_U");
    { Halide::RDom ru(0, n, "rgu_" + name);
      U(col, row) = zero;
      U(col, row) += W(ru, row) / Halide::max(S_raw(ru), tiny) *
                     Halide::cast(type, rank_f(ru) == col); }

    // Vt = permuted V^T: row k of Vt is V's column perm(k)
    Halide::Func Vt(name + "_Vt");
    { Halide::RDom rv(0, n, "rgv_" + name);
      Vt(col, row) = zero;
      Vt(col, row) += V(rv, col) * Halide::cast(type, rank_f(rv) == row); }

    return {U, S, Vt};
}

// -----------------------------------------------------------------------------
// Symmetric eigendecomposition via Jacobi (eigh)
// -----------------------------------------------------------------------------

/// @brief Result of a symmetric eigendecomposition
struct EighResult {
    Halide::Func eigenvalues;   ///< 1D, n eigenvalues (legacy form: unordered;
                                ///  typed overload: ASCENDING)
    Halide::Func eigenvectors;  ///< n×n, eigenvectors(col, row) — columns are
                                ///  e-vectors, paired with eigenvalues by index
};

/// @brief Symmetric eigendecomposition via classical Jacobi sweeps (LEGACY)
/// @param A        n×n symmetric matrix Func
/// @param n        Matrix size
/// @param n_sweeps Number of Jacobi sweeps (default 20)
/// @param name     Base name
/// @return EighResult {eigenvalues (1D), eigenvectors (n×n)}
///
/// After convergence the diagonal of the working matrix holds the eigenvalues
/// and V accumulates the eigenvectors (A = V · diag(λ) · V^T).
/// Note: creates O(n_sweeps * n²) stages; keep n ≤ 8.
///
/// LEGACY BEHAVIOR, kept bit-for-bit for existing callers: computes in f32
/// regardless of A's type, eigenvalues come out UNORDERED, eigenvector
/// columns carry no sign convention, and the trig rotation
/// (θ = ½·atan2(2·A_pq, A_pp − A_qq)) has a NaN derivative at diagonal
/// inputs (atan2(0, 0)). New code should use the typed overload below
/// (Type + eps parameters): type-generic, algebraic ζ/t/c/s rotations,
/// ascending eigenvalues, sign-normalized eigenvector columns, and a skip
/// guard that keeps reverse-mode derivatives finite.
inline EighResult eigh_jacobi(Halide::Func A, int n,
    int n_sweeps = 20, std::string const& name = "eigh")
{
    Halide::Var col("col"), row("row"), k("k");

    // Working matrix (symmetric, converges to diagonal)
    Halide::Func Aw(name + "_Aw0");
    Aw(col, row) = Halide::cast<float>(A(col, row));
    Aw.compute_root();

    // Eigenvector accumulation: starts as identity
    Halide::Func V(name + "_V0");
    V(col, row) = Halide::select(col == row, 1.0f, 0.0f);
    V.compute_root();

    for (int sweep = 0; sweep < n_sweeps; ++sweep) {
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const std::string tag = name
                    + "_s" + std::to_string(sweep)
                    + "p" + std::to_string(p)
                    + "q" + std::to_string(q);

                // Rotation angle: θ = 0.5·atan2(2·A[p,q], A[p,p]−A[q,q])
                // In Halide: A[row=p, col=q] = Aw(q, p)
                Halide::Expr theta = 0.5f * Halide::atan2(
                    2.0f * Aw(q, p), Aw(p, p) - Aw(q, q));

                Halide::Func cs_f(tag + "_cs");
                cs_f() = Halide::cos(theta);
                cs_f.compute_root();

                Halide::Func sn_f(tag + "_sn");
                sn_f() = Halide::sin(theta);
                sn_f.compute_root();

                // Row rotation: tmp = G^T · Aw
                Halide::Func tmp(tag + "_tmp");
                tmp(col, row) = Aw(col, row);
                { Halide::RDom rx1(0, n, "rx1_" + tag);
                  tmp(rx1, p) =  cs_f() * Aw(rx1, p) + sn_f() * Aw(rx1, q); }
                { Halide::RDom rx2(0, n, "rx2_" + tag);
                  tmp(rx2, q) = -sn_f() * Aw(rx2, p) + cs_f() * Aw(rx2, q); }
                tmp.compute_root();

                // Column rotation: Aw' = tmp · G
                Halide::Func Aw_next(tag + "_Aw");
                Aw_next(col, row) = tmp(col, row);
                { Halide::RDom ry1(0, n, "ry1_" + tag);
                  Aw_next(p, ry1) =  cs_f() * tmp(p, ry1) + sn_f() * tmp(q, ry1); }
                { Halide::RDom ry2(0, n, "ry2_" + tag);
                  Aw_next(q, ry2) = -sn_f() * tmp(p, ry2) + cs_f() * tmp(q, ry2); }
                Aw_next.compute_root();
                Aw = Aw_next;

                // Eigenvectors: V' = V · G  (right multiply)
                Halide::Func V_next(tag + "_V");
                V_next(col, row) = V(col, row);
                { Halide::RDom rv1(0, n, "rv1_" + tag);
                  V_next(p, rv1) =  cs_f() * V(p, rv1) + sn_f() * V(q, rv1); }
                { Halide::RDom rv2(0, n, "rv2_" + tag);
                  V_next(q, rv2) = -sn_f() * V(p, rv2) + cs_f() * V(q, rv2); }
                V_next.compute_root();
                V = V_next;
            }
        }
    }

    // Eigenvalues = diagonal of converged Aw
    Halide::Func evals(name + "_evals");
    evals(k) = Aw(k, k);

    return {evals, V};
}

/// @brief Symmetric eigendecomposition via ALGEBRAIC Jacobi sweeps (typed form)
/// @param A        n×n symmetric matrix Func (A(col, row))
/// @param n        Matrix size
/// @param n_sweeps Number of Jacobi sweeps
/// @param type     Element type to compute in (f32, f64, ...); A is cast to it
/// @param eps      Off-diagonal skip threshold, an Expr in `type` and ≪ 1
///                 (e.g. make_const(type, 1e-12)): |A_pq| < eps skips the
///                 (p,q) rotation. Besides saving work on converged pivots,
///                 this is what keeps reverse-mode derivatives FINITE at
///                 diagonal inputs (the legacy trig form NaNs there).
/// @param name     Base name. Default "eigh_t", deliberately distinct from
///                 the legacy default "eigh" so both forms can coexist in
///                 one pipeline without stage-name collisions.
/// @return EighResult:
///         - eigenvalues: 1D length n, sorted ASCENDING (stable)
///         - eigenvectors: n×n (col, row); column k is the unit eigenvector
///           of eigenvalues(k); each column is sign-normalized so its LAST
///           component (row n−1) is ≥ 0
///
/// Overloading note: this is a true overload of eigh_jacobi. Resolution
/// against the legacy form is unambiguous — the legacy form takes at most
/// 4 arguments with a std::string 4th, this form takes at least 5 with a
/// Halide::Type 4th, and neither type converts to the other.
///
/// Note: creates O(n_sweeps * n²) compute_root stages; keep n ≤ 8.
inline EighResult eigh_jacobi(Halide::Func A, int n, int n_sweeps,
    Halide::Type type, Halide::Expr eps, std::string const& name = "eigh_t")
{
    Halide::Var col("col"), row("row"), k("k");

    Halide::Expr zero    = Halide::Internal::make_zero(type);
    Halide::Expr one     = Halide::Internal::make_one(type);
    Halide::Expr two     = Halide::Internal::make_const(type, 2.0);
    Halide::Expr neg_one = Halide::Internal::make_const(type, -1.0);

    // Working matrix (symmetric, converges to diagonal)
    Halide::Func Aw(name + "_Aw0");
    Aw(col, row) = Halide::cast(type, A(col, row));
    Aw.compute_root();

    // Eigenvector accumulation: starts as identity
    Halide::Func V(name + "_V0");
    V(col, row) = Halide::select(col == row, one, zero);
    V.compute_root();

    // ROTATION ALGEBRA — algebraic ζ/t/c/s form (no trig; the legacy trig
    // θ = ½·atan2 form derives expensively under reverse-mode AD and its
    // atan2(0,0) derivative is NaN at diagonal inputs).
    //
    // J is identity except J[p][p]=c, J[p][q]=s, J[q][p]=−s, J[q][q]=c.
    // Update pairing: Aw' = Jᵀ·Aw·J while V accumulates V' = V·J, so
    // Aw_final = Vᵀ·A·V = Λ, i.e. A = V·Λ·Vᵀ — columns of V are the
    // eigenvectors, consistent with the gather below.
    //
    // Annihilation derivation (col p of J = c·e_p − s·e_q, col q of J =
    // s·e_p + c·e_q, and A_qp = A_pq by symmetry):
    //   A'_pq = (c·e_p − s·e_q)ᵀ A (s·e_p + c·e_q)
    //         = c·s·(A_pp − A_qq) + (c² − s²)·A_pq.
    // Setting A'_pq = 0 and dividing by c²·A_pq gives, with t = s/c and
    //   ζ = (A_qq − A_pp) / (2·A_pq):        t² + 2·ζ·t − 1 = 0.
    // The smaller-magnitude root t = sign(ζ)/(|ζ| + √(1+ζ²)) keeps
    // |t| ≤ 1 (rotation ≤ 45°, the convergent choice), then
    //   c = 1/√(1+t²),  s = t·c.
    // These signs annihilate ONLY with the update
    //   (p)' = c·(p) − s·(q),   (q)' = s·(p) + c·(q)
    // applied to rows and columns of Aw and to columns of V — the same
    // pairing svd_jacobi pins (the opposite-sign update is the historic
    // svd oscillation bug: γ never reduced, sweeps oscillate).
    // Diagonal effect of one rotation: A'_pp = A_pp − t·A_pq,
    // A'_qq = A_qq + t·A_pq.
    for (int sweep = 0; sweep < n_sweeps; ++sweep) {
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const std::string tag = name
                    + "_s" + std::to_string(sweep)
                    + "p" + std::to_string(p)
                    + "q" + std::to_string(q);

                // In Halide indexing Aw(col, row): A_pq = Aw(q, p)
                Halide::Expr apq = Aw(q, p);
                Halide::Expr app = Aw(p, p);
                Halide::Expr aqq = Aw(q, q);

                // SKIP GUARD via multiplied 0/1 indicators, NOT a select
                // around the division: reverse-mode AD of a select
                // synthesizes an f32 zero for the inactive arm, which
                // type-clashes with the f64 derivative of a division in
                // the arm (same rule as the svd_jacobi gathers). rot = 1
                // applies the rotation; rot = 0 yields the exact identity
                // (c = 1, s = 0) AND a zero adjoint through the rotation
                // branch: 0 × finite = 0, and the guarded denominator
                // below keeps the branch finite, so never inf × 0 = NaN.
                Halide::Expr rot = Halide::cast(type, Halide::abs(apq) >= eps);
                Halide::Expr skp = one - rot;

                // Guarded denominator: when skipping, +skp shifts it to
                // ≈1 (eps ≪ 1 ⇒ |2·A_pq + 1| ≥ 1 − 2·eps > 0); when
                // rotating, skp = 0 and it is exactly 2·A_pq with
                // |A_pq| ≥ eps. Finite everywhere, no select.
                Halide::Expr zeta = (aqq - app) / (two * apq + skp);
                Halide::Expr sgn_z = Halide::select(zeta >= zero, one, neg_one);
                Halide::Expr t_r = sgn_z
                    / (Halide::abs(zeta) + Halide::sqrt(one + zeta * zeta));

                Halide::Func cs_f(tag + "_cs");
                cs_f() = skp + rot * (one / Halide::sqrt(one + t_r * t_r));
                cs_f.compute_root();

                // rot² = rot and rot·skp = 0, so rot·cs_f()·t = rot·c_alg·t
                // — reuses the staged cosine instead of recomputing it.
                Halide::Func sn_f(tag + "_sn");
                sn_f() = rot * cs_f() * t_r;
                sn_f.compute_root();

                // Row rotation: tmp = Jᵀ · Aw
                //   row p' = c·row_p − s·row_q, row q' = s·row_p + c·row_q
                Halide::Func tmp(tag + "_tmp");
                tmp(col, row) = Aw(col, row);
                { Halide::RDom rx1(0, n, "rx1_" + tag);
                  tmp(rx1, p) = cs_f() * Aw(rx1, p) - sn_f() * Aw(rx1, q); }
                { Halide::RDom rx2(0, n, "rx2_" + tag);
                  tmp(rx2, q) = sn_f() * Aw(rx2, p) + cs_f() * Aw(rx2, q); }
                tmp.compute_root();

                // Column rotation: Aw' = tmp · J
                //   col p' = c·col_p − s·col_q, col q' = s·col_p + c·col_q
                Halide::Func Aw_next(tag + "_Aw");
                Aw_next(col, row) = tmp(col, row);
                { Halide::RDom ry1(0, n, "ry1_" + tag);
                  Aw_next(p, ry1) = cs_f() * tmp(p, ry1) - sn_f() * tmp(q, ry1); }
                { Halide::RDom ry2(0, n, "ry2_" + tag);
                  Aw_next(q, ry2) = sn_f() * tmp(p, ry2) + cs_f() * tmp(q, ry2); }
                Aw_next.compute_root();
                Aw = Aw_next;

                // Eigenvectors: V' = V · J (right multiply, same signs)
                Halide::Func V_next(tag + "_V");
                V_next(col, row) = V(col, row);
                { Halide::RDom rv1(0, n, "rv1_" + tag);
                  V_next(p, rv1) = cs_f() * V(p, rv1) - sn_f() * V(q, rv1); }
                { Halide::RDom rv2(0, n, "rv2_" + tag);
                  V_next(q, rv2) = sn_f() * V(p, rv2) + cs_f() * V(q, rv2); }
                V_next.compute_root();
                V = V_next;
            }
        }
    }

    // Raw eigenvalues = diagonal of converged Aw
    Halide::Func lam_raw(name + "_lraw");
    lam_raw(k) = Aw(k, k);
    lam_raw.compute_root();

    // ASCENDING order: stable rank of each diagonal entry, then gather.
    // Same AD-safe machinery as svd_jacobi's descending-S sort, with the
    // comparison flipped (< instead of >): rank(k) = number of entries
    // ordered strictly before k in a stable ascending sort.
    Halide::Func rank_f(name + "_rank");
    rank_f(k) = Halide::cast<int32_t>(0);
    { Halide::RDom rr(0, n, "rrank_" + name);
      rank_f(k) += Halide::select(
          lam_raw(rr) < lam_raw(k) || (lam_raw(rr) == lam_raw(k) && rr < k),
          Halide::cast<int32_t>(1), Halide::cast<int32_t>(0)); }
    rank_f.compute_root();

    // Gathers use MULTIPLIED 0/1 indicators, not select: reverse-mode AD
    // synthesizes a float32 zero for the inactive select branch, which
    // type-clashes with f64 derivatives inside the arm. The indicator
    // form derives cleanly in every type.
    Halide::Func evals(name + "_evals");
    { Halide::RDom rs(0, n, "rgl_" + name);
      evals(k) = zero;
      evals(k) += lam_raw(rs) * Halide::cast(type, rank_f(rs) == k); }
    evals.compute_root();

    // Eigenvector columns through the same permutation: output column k
    // is V's column perm(k). Still (col, row) — columns stay col-major
    // unit eigenvectors paired with evals by index.
    Halide::Func V_sorted(name + "_Vs");
    { Halide::RDom rv(0, n, "rgv_" + name);
      V_sorted(col, row) = zero;
      V_sorted(col, row) += V(rv, row) * Halide::cast(type, rank_f(rv) == col); }
    V_sorted.compute_root();

    // SIGN NORMALIZATION (deterministic output): after sorting, flip each
    // column so its LAST component (row n−1) is ≥ 0 — the cheapest
    // deterministic rule. Indicator form, no select:
    //   s = [V(k, n−1) ≥ 0]·2 − 1 ∈ {−1, +1}
    // A zero last component maps to +1 (0 ≥ 0 is true), so the rule is
    // total; ±v are both unit eigenvectors, so flipping preserves every
    // eigenpair property.
    Halide::Func sgn_f(name + "_sgn");
    sgn_f(k) = two * Halide::cast(type, V_sorted(k, n - 1) >= zero) - one;
    sgn_f.compute_root();

    Halide::Func evecs(name + "_evecs");
    evecs(col, row) = V_sorted(col, row) * sgn_f(col);

    return {evals, evecs};
}

// -----------------------------------------------------------------------------
// Solve linear system and least-squares
// -----------------------------------------------------------------------------

/// @brief Solve the square linear system A·x = b via QR decomposition
/// @param A  n×n matrix Func
/// @param b  1D n-vector Func
/// @param n  System size
/// @param name Base name
/// @param eps Threaded through to the qr_gs column-norm guard and the
///            back_sub pivot guard. 0 keeps the unguarded behavior.
/// @return 1D solution Func x(i) such that A·x ≈ b
///
/// Computes in A's element type (f32, f64, ...); b must match.
inline Halide::Func solve(Halide::Func A, Halide::Func b, int n,
    std::string const& name = "solve", float eps = 0.0f)
{
    Halide::Type type = A.types()[0];
    Halide::Expr zero = Halide::Internal::make_const(type, 0.0);

    auto [Q, R] = qr_gs(A, n, n, name + "_qr", eps);
    Q.compute_root();
    R.compute_root();

    // y = Q^T · b  →  y[i] = Σ_r Q(col=i, row=r) * b(r)
    Halide::Var i("i");
    Halide::RDom r_qtb(0, n, "rqtb");
    Halide::Func y(name + "_y");
    y(i) = zero;
    y(i) += Q(i, r_qtb) * b(r_qtb);
    y.compute_root();

    return back_sub(R, y, n, name + "_bs", eps);
}

/// @brief Least-squares solution to A·x ≈ b (minimises ||A·x − b||₂)
/// @param A  m×n matrix Func (m ≥ n)
/// @param b  1D m-vector Func
/// @param m  Number of rows
/// @param n  Number of columns
/// @param name Base name
/// @param eps Threaded through to the qr_gs column-norm guard and the
///            back_sub pivot guard. 0 keeps the unguarded behavior.
/// @return 1D solution Func x(i) of length n
///
/// Computes in A's element type (f32, f64, ...); b must match.
inline Halide::Func lstsq(Halide::Func A, Halide::Func b, int m, int n,
    std::string const& name = "lstsq", float eps = 0.0f)
{
    nh_require(m >= n, "lstsq: requires m >= n, got m=%d n=%d", m, n);

    auto [Q, R] = qr_gs(A, m, n, name + "_qr", eps);
    Q.compute_root();
    R.compute_root();

    // y = Q^T · b  (n-vector)
    Halide::Var i("i");
    Halide::RDom r_qtb(0, m, "rqtb_ls");
    Halide::Func y(name + "_y");
    y(i) = Halide::cast(A.types()[0], 0);
    y(i) += Q(i, r_qtb) * b(r_qtb);
    y.compute_root();

    return back_sub(R, y, n, name + "_bs", eps);
}

// -----------------------------------------------------------------------------
// Pseudo-inverse via SVD
// -----------------------------------------------------------------------------

/// @brief Moore-Penrose pseudo-inverse: A⁺ = V · diag(σ⁺) · U^T
/// @param A    m×n matrix Func
/// @param m    Number of rows
/// @param n    Number of columns
/// @param tol  Threshold below which singular values are treated as zero
///             (compared in A's element type)
/// @param n_sweeps SVD Jacobi sweeps
/// @param name Base name
/// @return n×m pseudo-inverse Func pinv(col=j, row=i) = A⁺[i,j]
///
/// Computes in A's element type (f32, f64, ...).
inline Halide::Func pinv(Halide::Func A, int m, int n,
    float tol = 1e-10f, int n_sweeps = 10,
    std::string const& name = "pinv")
{
    Halide::Type type = A.types()[0];
    Halide::Expr zero  = Halide::Internal::make_const(type, 0.0);
    Halide::Expr one   = Halide::Internal::make_const(type, 1.0);
    Halide::Expr tiny  = Halide::Internal::make_const(type, 1e-30);
    Halide::Expr tol_e = Halide::Internal::make_const(type, static_cast<double>(tol));

    auto svd = svd_jacobi(A, m, n, n_sweeps, name + "_svd");
    svd.U.compute_root();
    svd.S.compute_root();
    svd.Vt.compute_root();

    // S⁺[k] = 1/S[k] if S[k] > tol, else 0 — written as a MULTIPLIED 0/1
    // indicator with a floored divisor, NOT select(S > tol, 1/S, 0):
    // reverse-mode AD synthesizes a float32 zero for the inactive select
    // branch, which type-clashes with the f64 derivative of the division
    // inside the arm. The indicator form derives cleanly in every type
    // (same pattern as the svd_jacobi gathers above).
    Halide::Var k("k");
    Halide::Func Sinv(name + "_Sinv");
    Sinv(k) = (one / Halide::max(svd.S(k), tiny))
              * Halide::cast(type, svd.S(k) > tol_e);
    Sinv.compute_root();

    // pinv(col=j, row=i) = Σ_k V[i,k] * S⁺[k] * U[j,k]
    //   V[i,k]  = Vt^T[i,k] = Vt[k,i] = Vt(col=i, row=k) = Vt(i, k)
    //   U^T[k,j] = U[j,k]   = U(col=k, row=j) = U(k, j)
    Halide::Var pi("pi"), pj("pj");
    Halide::RDom rk(0, n, "rk_pinv");
    Halide::Func ret(name);
    ret(pj, pi) = zero;
    ret(pj, pi) += svd.Vt(pi, rk) * Sinv(rk) * svd.U(rk, pj);
    return ret;
}

// -----------------------------------------------------------------------------
// General determinant via LU decomposition (no pivoting)
// -----------------------------------------------------------------------------

/// @brief Compute det(A) for an n×n matrix via Doolittle LU (no pivoting)
/// @param A  n×n matrix Func
/// @param n  Matrix size
/// @param name Base name
/// @return 0D scalar Func; realize with Buffer<float>::make_scalar()
///
/// Warning: no partial pivoting — may give inaccurate results for
/// ill-conditioned or nearly-singular matrices.
inline Halide::Func det_lu(Halide::Func A, int n,
    std::string const& name = "det_lu")
{
    Halide::Var col("col"), row("row");

    Halide::Func W(name + "_W0");
    W(col, row) = Halide::cast<float>(A(col, row));
    W.compute_root();

    for (int k = 0; k < n - 1; ++k) {
        std::string sk = std::to_string(k);

        // Store multipliers in column k below the diagonal
        Halide::Func W_mul(name + "_mul" + sk);
        W_mul(col, row) = W(col, row);
        {
            Halide::RDom ri(k + 1, n - k - 1, "ri_mul" + sk);
            // W[row=ri, col=k] /= W[k,k]
            W_mul(k, ri) = W(k, ri) / W(k, k);
        }
        W_mul.compute_root();

        // Eliminate submatrix: W[i,j] -= W[i,k]*W[k,j]  (i,j > k)
        Halide::Func W_sub(name + "_sub" + sk);
        W_sub(col, row) = W_mul(col, row);
        {
            Halide::RDom r2(k + 1, n - k - 1, k + 1, n - k - 1, "rs2" + sk);
            W_sub(r2.x, r2.y) -= W_mul(k, r2.y) * W_mul(r2.x, k);
        }
        W_sub.compute_root();
        W = W_sub;
    }

    // det = product of upper-triangular diagonal
    Halide::RDom rk(0, n, "rk_det");
    Halide::Func det_f(name + "_d");
    det_f() = 1.0f;
    det_f() *= W(rk, rk);

    return det_f;
}

// -----------------------------------------------------------------------------
// matrix_rank — count singular values above threshold
// -----------------------------------------------------------------------------

/// @brief Estimate the rank of matrix A via SVD singular value thresholding
/// @param A        m×n matrix Func (A(col, row))
/// @param m        Number of rows
/// @param n        Number of columns (m ≥ n required)
/// @param tol      Singular values ≤ tol are treated as zero (default 1e-10)
/// @param n_sweeps Jacobi SVD sweeps (default 10)
/// @param name     Base name
/// @return 0D scalar Func (float); realize with Buffer<float>::make_scalar()
inline Halide::Func matrix_rank(Halide::Func A, int m, int n,
    float tol = 1e-10f, int n_sweeps = 10,
    std::string const& name = "matrix_rank")
{
    auto svd = svd_jacobi(A, m, n, n_sweeps, name + "_svd");
    svd.S.compute_root();

    Halide::RDom rk(0, n, "rk_mr");
    Halide::Func ret(name);
    ret() = 0.0f;
    ret() += Halide::select(svd.S(rk) > tol, 1.0f, 0.0f);
    return ret;
}

// -----------------------------------------------------------------------------
// cond — condition number (ratio of largest to smallest singular value)
// -----------------------------------------------------------------------------

/// @brief Condition number of matrix A: σ_max / σ_min
/// @param A        m×n matrix Func (m ≥ n)
/// @param m        Number of rows
/// @param n        Number of columns
/// @param n_sweeps Jacobi SVD sweeps (default 10)
/// @param name     Base name
/// @return 0D scalar Func; realize with Buffer<float>::make_scalar()
inline Halide::Func cond(Halide::Func A, int m, int n,
    int n_sweeps = 10, std::string const& name = "cond")
{
    auto svd = svd_jacobi(A, m, n, n_sweeps, name + "_svd");
    svd.S.compute_root();

    Halide::Var k("k_cond");

    Halide::Func s_max(name + "_max");
    s_max() = svd.S(0);
    { Halide::RDom rk(1, n - 1, "rk_cmax");
      s_max() = Halide::max(s_max(), svd.S(rk)); }
    s_max.compute_root();

    Halide::Func s_min(name + "_min");
    s_min() = svd.S(0);
    { Halide::RDom rk(1, n - 1, "rk_cmin");
      s_min() = Halide::min(s_min(), svd.S(rk)); }
    s_min.compute_root();

    Halide::Func ret(name);
    ret() = s_max() / s_min();
    return ret;
}

// -----------------------------------------------------------------------------
// slogdet — sign and log of determinant
// -----------------------------------------------------------------------------

/// @brief Result of slogdet: sign * exp(logabsdet) = det(A)
struct SlogdetResult {
    Halide::Func sign;       ///< 0D scalar: +1, -1, or 0
    Halide::Func logabsdet;  ///< 0D scalar: log|det(A)|
};

/// @brief Compute (sign, log|det|) of an n×n matrix via LU decomposition
/// @param A    n×n matrix Func
/// @param n    Matrix size
/// @param name Base name
/// @return SlogdetResult; realize each with Buffer<float>::make_scalar()
///
/// Uses Doolittle LU (no pivoting) — inaccurate for ill-conditioned matrices.
/// sign = product of sgn(U_ii); logabsdet = sum(log|U_ii|)
inline SlogdetResult slogdet(Halide::Func A, int n,
    std::string const& name = "slogdet")
{
    Halide::Var col("col"), row("row");

    Halide::Func W(name + "_W0");
    W(col, row) = Halide::cast<float>(A(col, row));
    W.compute_root();

    for (int k = 0; k < n - 1; ++k) {
        std::string sk = std::to_string(k);

        Halide::Func W_mul(name + "_mul" + sk);
        W_mul(col, row) = W(col, row);
        { Halide::RDom ri(k + 1, n - k - 1, "ri_sm" + sk);
          W_mul(k, ri) = W(k, ri) / W(k, k); }
        W_mul.compute_root();

        Halide::Func W_sub(name + "_sub" + sk);
        W_sub(col, row) = W_mul(col, row);
        { Halide::RDom r2(k + 1, n - k - 1, k + 1, n - k - 1, "rs2_sm" + sk);
          W_sub(r2.x, r2.y) -= W_mul(k, r2.y) * W_mul(r2.x, k); }
        W_sub.compute_root();
        W = W_sub;
    }

    // sign = product of sgn(U_ii); logabsdet = sum(log|U_ii|)
    Halide::RDom rk(0, n, "rk_sd");
    Halide::Func sign_f(name + "_sign");
    sign_f() = 1.0f;
    sign_f() *= Halide::select(W(rk, rk) >= 0.0f, 1.0f, -1.0f);

    Halide::RDom rk2(0, n, "rk2_sd");
    Halide::Func logdet_f(name + "_logdet");
    logdet_f() = 0.0f;
    logdet_f() += Halide::log(Halide::abs(W(rk2, rk2)));

    return { sign_f, logdet_f };
}

// -----------------------------------------------------------------------------
// eig — general eigenvalues
// -----------------------------------------------------------------------------

/// @brief Result of a general eigendecomposition: real and imaginary parts
struct EigResult {
    Halide::Func real;  ///< 1D, n real parts of eigenvalues
    Halide::Func imag;  ///< 1D, n imaginary parts (nonzero for complex-conjugate pairs)
};

/// @brief Exact eigenvalues of a 2×2 real matrix (may be complex)
/// @param A    2×2 Func
/// @param name Base name
/// @return EigResult: eigenvalues (λ₀, λ₁) sorted so Im(λ₀) ≥ 0
///
/// Uses the analytic formula: λ = (tr ± √(tr²−4det)) / 2
/// When the discriminant is negative the eigenvalues are complex conjugates.
inline EigResult eig2x2(Halide::Func A, std::string const& name = "eig2")
{
    Halide::Expr a = A(0, 0), b = A(1, 0), c = A(0, 1), d = A(1, 1);
    Halide::Expr tr   = a + d;
    Halide::Expr det4 = 4.0f * (a * d - b * c);
    Halide::Expr disc = tr * tr - det4;           // discriminant
    Halide::Expr sqD  = Halide::sqrt(Halide::abs(disc));
    Halide::Expr half = 0.5f;

    Halide::Var k("k");
    Halide::Func re(name + "_re"), im(name + "_im");

    // Real eigenvalue pair (disc ≥ 0): (tr ± sqD) / 2
    // Complex conjugate pair (disc < 0): tr/2 ± i*sqD/2
    re(k) = tr * half + Halide::select(disc >= 0.0f,
        Halide::select(k == 0,  sqD * half, -sqD * half),
        0.0f);
    im(k) = Halide::select(disc >= 0.0f,
        0.0f,
        Halide::select(k == 0,  sqD * half, -sqD * half));

    return { re, im };
}

/// @brief General eigenvalues of an n×n real matrix via QR iteration
/// @param A        n×n matrix Func
/// @param n        Matrix size (recommend n ≤ 6 for reasonable compile time)
/// @param n_iter   Number of QR iterations (default 40)
/// @param name     Base name
/// @return EigResult with real and zero imaginary parts
///
/// Algorithm: repeatedly factor A = QR then set A ← RQ.
/// Converges to upper quasi-triangular (real Schur) form; diagonal ≈ eigenvalues.
/// Complex conjugate pairs produce 2×2 diagonal blocks — their real parts are
/// returned as two identical values; imaginary parts are set to 0 in this
/// implementation. Use eig2x2() for exact complex eigenvalues of 2×2 matrices.
inline EigResult eig_qr(Halide::Func A, int n,
    int n_iter = 40, std::string const& name = "eig")
{
    // Special-case the 2×2 exact formula
    if (n == 2) return eig2x2(A, name);

    Halide::Var col("col"), row("row");
    Halide::Func Ak(name + "_A0");
    Ak(col, row) = Halide::cast<float>(A(col, row));
    Ak.compute_root();

    shape_t snn = { n, n };

    for (int iter = 0; iter < n_iter; ++iter) {
        const std::string tag = name + "_i" + std::to_string(iter);
        auto [Q, R] = qr_gs(Ak, n, n, tag + "_qr");
        Q.compute_root();
        R.compute_root();
        Halide::Func Anext = matmul(R, snn, Q, snn, tag + "_RQ");
        Anext.compute_root();
        Ak = Anext;
    }

    // Extract diagonal as eigenvalues
    Halide::Var k("k");
    Halide::Func re(name + "_re"), im(name + "_im");
    re(k) = Ak(k, k);
    im(k) = 0.0f;
    return { re, im };
}

NS_NUM_HALIDE_END
