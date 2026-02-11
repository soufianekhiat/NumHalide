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
	nh_require(nullptr, shape_a.rank == 2 && shape_b.rank == 2,
		"matmul requires 2D matrices, got ranks %d and %d", shape_a.rank, shape_b.rank);

	int M = shape_a.extents[0];  // rows of a
	int K = shape_a.extents[1];  // cols of a = rows of b
	int N = shape_b.extents[1];  // cols of b

	nh_require(nullptr, shape_a.extents[1] == shape_b.extents[0],
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
	nh_require(nullptr, a.rank == 2 && b.rank == 2,
		"matmul requires 2D matrices, got ranks %d and %d", a.rank, b.rank);
	nh_require(nullptr, a.extents[1] == b.extents[0],
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

	nh_require(nullptr, shape_a.rank >= 2 && shape_a.rank <= 3,
		"batched_matmul requires 2D or 3D input, got rank %d", shape_a.rank);
	nh_require(nullptr, shape_b.rank >= 2 && shape_b.rank <= 3,
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
			nh_require(nullptr, shape_b.extents[0] == batch,
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

	nh_require(nullptr, K_a == K_b,
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
			nh_require(nullptr, b.extents[0] == batch, "Batch dimensions must match");
		} else {
			batch = b.extents[0];
		}
		K_b = b.extents[1];
		N = b.extents[2];
	} else {
		K_b = b.extents[0];
		N = b.extents[1];
	}

	nh_require(nullptr, K_a == K_b, "Inner dimensions must match");

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
	nh_require(nullptr, shape_a.rank == 1 && shape_b.rank == 1,
		"dot requires 1D vectors, got ranks %d and %d", shape_a.rank, shape_b.rank);
	nh_require(nullptr, shape_a.extents[0] == shape_b.extents[0],
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
	nh_require(nullptr, shape_a.rank == 1 && shape_b.rank == 2,
		"dot requires 1D vector and 2D matrix, got ranks %d and %d", shape_a.rank, shape_b.rank);
	nh_require(nullptr, shape_a.extents[0] == shape_b.extents[0],
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
	nh_require(nullptr, shape_a.rank == 1 && shape_b.rank == 1,
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
	nh_require(nullptr, a.rank == 1 && b.rank == 1,
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
	nh_require(nullptr, shape_mat.rank == 2 && shape_vec.rank == 1,
		"matvec requires 2D matrix and 1D vector, got ranks %d and %d", shape_mat.rank, shape_vec.rank);
	nh_require(nullptr, shape_mat.extents[1] == shape_vec.extents[0],
		"matvec: matrix columns must match vector length: %d vs %d", shape_mat.extents[1], shape_vec.extents[0]);

	int M = shape_mat.extents[0];
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
	nh_require(nullptr, mat.rank == 2 && vec.rank == 1,
		"matvec requires 2D matrix and 1D vector");
	nh_require(nullptr, mat.extents[1] == vec.extents[0],
		"matvec: matrix columns must match vector length");
	return shape_t{ mat.extents[0] };
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
	nh_require(nullptr, shape_mat.rank == 2,
		"trace requires 2D matrix, got rank %d", shape_mat.rank);
	nh_require(nullptr, shape_mat.extents[0] == shape_mat.extents[1],
		"trace requires square matrix, got %dx%d", shape_mat.extents[0], shape_mat.extents[1]);

	int N = shape_mat.extents[0];

	Halide::Func ret(name);
	Halide::Var x("x");
	Halide::RDom i(0, N, "i");

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
	nh_require(nullptr, shape_mat.rank == 2,
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
	nh_require(nullptr, shape_vec.rank == 1,
		"diag_matrix requires 1D vector, got rank %d", shape_vec.rank);

	Halide::Func ret(name);
	Halide::Var x("x"), y("y");

	ret(x, y) = Halide::select(x == y, vec(x), Halide::cast(vec.types()[0], 0));

	return ret;
}

/// @brief Infer output shape for diag
inline shape_t infer_diag(const shape_t& mat) {
	nh_require(nullptr, mat.rank == 2, "diag requires 2D matrix");
	int min_dim = std::min(mat.extents[0], mat.extents[1]);
	return shape_t{ min_dim };
}

/// @brief Infer output shape for diag_matrix
inline shape_t infer_diag_matrix(const shape_t& vec) {
	nh_require(nullptr, vec.rank == 1, "diag_matrix requires 1D vector");
	return shape_t{ vec.extents[0], vec.extents[0] };
}

NS_NUM_HALIDE_END
