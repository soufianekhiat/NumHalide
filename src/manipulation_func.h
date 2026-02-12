/// @file manipulation_func.h
/// @brief Manipulation functions for Halide::Func objects
///
/// Provides: flatten, reshape, concat, vstack, hstack, dstack

#pragma once

#include "common.h"
#include "numhalide.h"
#include "shape.h"

NS_NUM_HALIDE_BEGIN

/// @brief Flatten a multidimensional Func into 1D
/// @param in Input Func
/// @param sizes Dimension sizes
/// @param name Function name
/// @return Flattened 1D Func
///
/// Usage:
///   Var x, y;
///   Func in; in(x, y) = x + y;
///   Func flat = flatten(in, {5, 4});
inline
Halide::Func flatten(Halide::Func in, std::vector<Halide::Expr> const& sizes, std::string const& name = "flatten")
{
    Halide::Func ret(name);
    Halide::Var x;
    std::vector<Halide::Expr> aArgs;
    index_to_args(aArgs, x, sizes);
    ret(x) = in(aArgs);
    return ret;
}

/// @brief Reshape a Func to new dimensions
/// @param in Input Func
/// @param sizes Original dimension sizes
/// @param new_sizes New dimension sizes
/// @param name Function name
/// @return Reshaped Func
///
/// Usage:
///   Func in; // 5x4 array
///   Func reshaped = reshape(in, {5,4}, {4,5});
inline
Halide::Func reshape(Halide::Func in, std::vector<Halide::Expr> const& sizes, std::vector<Halide::Expr> const& new_sizes, std::string const& name = "reshape")
{
    Halide::Func ret(name);
    std::vector<Halide::Var> vars;
    vars.resize(new_sizes.size());

    std::vector<Halide::Expr> varsVal;
    varsVal.reserve(sizes.size() + new_sizes.size());
    Halide::Expr prod0 = 1;
    for (Halide::Expr e : sizes) { prod0 *= e; varsVal.push_back(e); }
    Halide::Expr prod1 = 1;
    for (Halide::Expr e : new_sizes) { prod1 *= e; varsVal.push_back(e); }
    Halide::require(prod0 == prod1, varsVal);

    std::vector<Halide::Expr> exprVars;
    exprVars.resize(vars.size());
    std::transform(vars.begin(), vars.end(), exprVars.begin(), [](Halide::Var v){ return (Halide::Expr)v; });
    Halide::Expr idx = args_to_index(exprVars, new_sizes);

    Halide::Func flat = flatten(in, sizes);
    ret(vars) = flat(idx);
    return ret;
}

/// @brief Concatenate two Funcs along an axis
/// @param a First Func
/// @param shape_a Shape of first Func
/// @param b Second Func
/// @param shape_b Shape of second Func
/// @param axis Axis to concatenate along
/// @param name Function name
/// @return Concatenated Func
inline
Halide::Func concat(Halide::Func a, const shape_t& shape_a, Halide::Func b, const shape_t& shape_b, int axis, std::string const& name = "concat")
{
    int norm_axis = normalized_axis(axis, shape_a.rank);
    nh_require(nullptr, check_same_except(shape_a, shape_b, norm_axis),
        "Shapes %s and %s mismatch for concat at axis %d",
        shape_to_string(shape_a).c_str(), shape_to_string(shape_b).c_str(), axis);

    Halide::Func ret(name);
    std::vector<Halide::Var> vars;
    for (int i = 0; i < shape_a.rank; ++i) {
        vars.push_back(Halide::Var());
    }

    // norm_axis is the index in shape (0..rank-1).
    // The corresponding Halide Var index is rank - 1 - norm_axis.
    int var_idx = shape_a.rank - 1 - norm_axis;
    Halide::Expr split_point = shape_a.extents[norm_axis];
    Halide::Var split_var = vars[var_idx];

    std::vector<Halide::Expr> args_b;
    // b expects args in order [dim_rank-1, ..., dim_0] (inner to outer)
    // We iterate k from rank-1 down to 0.
    for (int k = shape_a.rank - 1; k >= 0; --k) {
        int current_var_idx = shape_a.rank - 1 - k;
        if (k == norm_axis) {
            args_b.push_back(vars[current_var_idx] - split_point);
        } else {
            args_b.push_back(vars[current_var_idx]);
        }
    }

    // Ensure boolean condition for select
    Halide::Expr cond = split_var < split_point;
    // Halide::select requires Expr arguments, a(vars) and b(args_b) return FuncRef/Expr
    ret(vars) = Halide::select(cond, a(vars), b(args_b));
    return ret;
}

/// @brief Stack arrays vertically (along axis 0)
inline
Halide::Func vstack(Halide::Func a, const shape_t& shape_a, Halide::Func b, const shape_t& shape_b, std::string const& name = "vstack")
{
    if (shape_a.rank == 1 && shape_b.rank == 1) {
        nh_require(nullptr, shape_a.extents[0] == shape_b.extents[0],
            "Shapes %s and %s mismatch for vstack (dim 1 must match)",
            shape_to_string(shape_a).c_str(), shape_to_string(shape_b).c_str());
        Halide::Func ret(name);
        Halide::Var x("x"), y("y");
        // Row 0 from a, Row 1 from b
        ret(x, y) = Halide::select(y < 1, a(x), b(x));
        return ret;
    }
    return concat(a, shape_a, b, shape_b, 0, name);
}

/// @brief Stack arrays horizontally (along axis 1)
inline
Halide::Func hstack(Halide::Func a, const shape_t& shape_a, Halide::Func b, const shape_t& shape_b, std::string const& name = "hstack")
{
    // Special case for 2D arrays: stack horizontally (columns)
    if (shape_a.rank == 2 && shape_b.rank == 2 && shape_a.extents[0] == shape_b.extents[0]) {
        // Ensure same number of rows
        nh_require(nullptr, shape_a.extents[0] == shape_b.extents[0],
            "Shapes %s and %s mismatch for hstack (rows must match)",
            shape_to_string(shape_a).c_str(), shape_to_string(shape_b).c_str());
        Halide::Func ret(name);
        Halide::Var y("y"), x("x");
        Halide::Expr split_point = shape_a.extents[1]; // columns of a
        // Row index y, column index x
        ret(x, y) = Halide::select(x < split_point, a(x, y), b(x - split_point, y));
        return ret;
    }
    return concat(a, shape_a, b, shape_b, 1, name);
}

/// @brief Stack arrays depth‑wise (along axis 2)
inline
Halide::Func dstack(Halide::Func a, const shape_t& shape_a, Halide::Func b, const shape_t& shape_b, std::string const& name = "dstack")
{
    return concat(a, shape_a, b, shape_b, 2, name);
}

// -----------------------------------------------------------------------------
// Slicing Operations
// -----------------------------------------------------------------------------

/// @brief Extract a slice of a Func along an axis
/// @param f Input Func
/// @param in_shape Shape of input
/// @param axis Axis to slice along
/// @param start Start index (can be negative)
/// @param stop Stop index exclusive (can be negative)
/// @param step Step size (default 1)
/// @param name Function name
/// @return Sliced Func
///
/// Usage:
///   // For a 3x4 matrix, get rows 0-2 with step 1
///   Func sliced = slice(mat, {3, 4}, 0, 0, 2, 1);
inline
Halide::Func slice(Halide::Func f, const shape_t& in_shape, int axis, int start, int stop, int step = 1, std::string const& name = "slice")
{
	int norm_axis = normalized_axis(axis, in_shape.rank);
	int extent = in_shape.extents[norm_axis];

	// Normalize negative indices
	if (start < 0) start += extent;
	if (stop < 0) stop += extent;

	// Clamp to valid range
	start = std::max(0, std::min(start, extent));
	stop = std::max(0, std::min(stop, extent));

	nh_require(nullptr, step != 0, "Slice step cannot be zero");

	Halide::Func ret(name);
	std::vector<Halide::Var> out_vars;
	for (int i = 0; i < in_shape.rank; ++i) {
		out_vars.push_back(Halide::Var());
	}

	// Build input arguments
	// Shape convention: extents[0] is outermost, extents[rank-1] is innermost
	// Halide convention: vars[0] is innermost, vars[rank-1] is outermost
	std::vector<Halide::Expr> in_args;
	for (int shape_dim = in_shape.rank - 1; shape_dim >= 0; --shape_dim) {
		int var_idx = in_shape.rank - 1 - shape_dim;
		if (shape_dim == norm_axis) {
			// Map output index to input index
			in_args.push_back(start + out_vars[var_idx] * step);
		} else {
			in_args.push_back(out_vars[var_idx]);
		}
	}

	ret(out_vars) = f(in_args);
	return ret;
}

/// @brief Take elements from a Func along an axis using indices
/// @param f Input Func
/// @param in_shape Shape of input
/// @param indices Vector of indices to take
/// @param axis Axis to take along
/// @param name Function name
/// @return Func with gathered elements
///
/// Usage:
///   // For a 4x3 matrix, take rows 0 and 2
///   Func taken = take(mat, {4, 3}, {0, 2}, 0);
inline
Halide::Func take(Halide::Func f, const shape_t& in_shape, const std::vector<int>& indices, int axis, std::string const& name = "take")
{
	int norm_axis = normalized_axis(axis, in_shape.rank);
	int extent = in_shape.extents[norm_axis];
	int n_indices = static_cast<int>(indices.size());

	// Validate indices
	for (int idx : indices) {
		int norm_idx = idx < 0 ? idx + extent : idx;
		nh_require(nullptr, norm_idx >= 0 && norm_idx < extent,
			"Index %d out of bounds for axis %d with size %d", idx, axis, extent);
	}

	// Build a lookup table for indices
	Halide::Func idx_table("idx_table");
	Halide::Var i;
	Halide::Expr idx_expr = 0;
	for (int j = n_indices - 1; j >= 0; --j) {
		int norm_idx = indices[j] < 0 ? indices[j] + extent : indices[j];
		idx_expr = Halide::select(i == j, norm_idx, idx_expr);
	}
	idx_table(i) = idx_expr;

	Halide::Func ret(name);
	std::vector<Halide::Var> out_vars;
	for (int d = 0; d < in_shape.rank; ++d) {
		out_vars.push_back(Halide::Var());
	}

	// Build input arguments
	std::vector<Halide::Expr> in_args;
	for (int shape_dim = in_shape.rank - 1; shape_dim >= 0; --shape_dim) {
		int var_idx = in_shape.rank - 1 - shape_dim;
		if (shape_dim == norm_axis) {
			// Look up the actual index from the table
			in_args.push_back(idx_table(out_vars[var_idx]));
		} else {
			in_args.push_back(out_vars[var_idx]);
		}
	}

	ret(out_vars) = f(in_args);
	return ret;
}

/// @brief Infer output shape for take operation
inline shape_t infer_take(const shape_t& in, int axis, int n_indices) {
	int norm_axis = normalized_axis(axis, in.rank);
	shape_t res = in;
	res.extents[norm_axis] = n_indices;
	return res;
}

// -----------------------------------------------------------------------------
// Transpose and Axis Reordering
// -----------------------------------------------------------------------------

/// @brief Transpose (permute dimensions of) a Func
/// @param f Input Func
/// @param in_shape Shape of input
/// @param axes Permutation of axes. axes[i] specifies which input axis goes to output position i.
/// @param name Function name
/// @return Transposed Func
///
/// Usage:
///   // For a 2x3 matrix, transpose to 3x2
///   Func transposed = transpose(mat, {2, 3}, {1, 0});
inline
Halide::Func transpose(Halide::Func f, const shape_t& in_shape, const std::vector<int>& axes, std::string const& name = "transpose")
{
	nh_require(nullptr, static_cast<int>(axes.size()) == in_shape.rank,
		"Transpose axes length %d does not match rank %d",
		static_cast<int>(axes.size()), in_shape.rank);

	// Normalize axes and check for valid permutation
	std::vector<int> norm_axes;
	std::vector<bool> seen(in_shape.rank, false);
	for (int ax : axes) {
		int norm = normalized_axis(ax, in_shape.rank);
		nh_require(nullptr, !seen[norm], "Duplicate axis %d in transpose", ax);
		seen[norm] = true;
		norm_axes.push_back(norm);
	}

	Halide::Func ret(name);
	std::vector<Halide::Var> out_vars;
	for (int i = 0; i < in_shape.rank; ++i) {
		out_vars.push_back(Halide::Var());
	}

	// Build input arguments
	// out_shape.extents[out_dim] = in_shape.extents[axes[out_dim]]
	// For Halide indexing:
	// - Shape dim 0 is outermost, shape dim rank-1 is innermost
	// - Halide var 0 is innermost, var rank-1 is outermost
	//
	// We need to map output vars to input args
	// in_args are ordered: [in_dim_rank-1, ..., in_dim_0] (innermost to outermost input dims)
	std::vector<Halide::Expr> in_args(in_shape.rank);

	for (int out_shape_dim = 0; out_shape_dim < in_shape.rank; ++out_shape_dim) {
		int in_shape_dim = norm_axes[out_shape_dim];
		// out_shape_dim -> out_var at index (rank - 1 - out_shape_dim)
		int out_var_idx = in_shape.rank - 1 - out_shape_dim;
		// in_shape_dim -> in_arg at index (rank - 1 - in_shape_dim)
		int in_arg_idx = in_shape.rank - 1 - in_shape_dim;
		in_args[in_arg_idx] = out_vars[out_var_idx];
	}

	ret(out_vars) = f(in_args);
	return ret;
}

/// @brief Transpose a 2D Func (swap rows and columns)
/// @param f Input Func (2D)
/// @param in_shape Shape of input (must be rank 2)
/// @param name Function name
/// @return Transposed Func
inline
Halide::Func transpose(Halide::Func f, const shape_t& in_shape, std::string const& name = "transpose")
{
	nh_require(nullptr, in_shape.rank == 2, "2D transpose requires rank 2, got %d", in_shape.rank);
	return transpose(f, in_shape, {1, 0}, name);
}

/// @brief Move an axis from one position to another
/// @param f Input Func
/// @param in_shape Shape of input
/// @param src Source axis position
/// @param dst Destination axis position
/// @param name Function name
/// @return Func with axis moved
///
/// Usage:
///   // Move axis 0 to position 2 for a 3D tensor
///   Func moved = moveaxis(tensor, {2, 3, 4}, 0, 2);
inline
Halide::Func moveaxis(Halide::Func f, const shape_t& in_shape, int src, int dst, std::string const& name = "moveaxis")
{
	std::vector<int> perm = compute_moveaxis_perm(in_shape.rank, src, dst);
	return transpose(f, in_shape, perm, name);
}

/// @brief Infer output shape for moveaxis
inline shape_t infer_moveaxis(const shape_t& in, int src, int dst) {
	std::vector<int> perm = compute_moveaxis_perm(in.rank, src, dst);
	return infer_transpose(in, perm);
}

// -----------------------------------------------------------------------------
// Additional Dimension Manipulation
// -----------------------------------------------------------------------------

/// @brief Add a new axis of size 1 at the specified position
/// @param f Input Func
/// @param in_shape Shape of input
/// @param axis Position to insert new axis
/// @param name Function name
/// @return Func with expanded dimension
inline
Halide::Func expand_dims(Halide::Func f, const shape_t& in_shape, int axis, std::string const& name = "expand_dims")
{
	int new_rank = in_shape.rank + 1;
	int norm_axis = axis < 0 ? axis + new_rank : axis;
	nh_require(nullptr, norm_axis >= 0 && norm_axis <= in_shape.rank,
		"Axis %d out of bounds for expand_dims with rank %d", axis, in_shape.rank);

	Halide::Func ret(name);
	std::vector<Halide::Var> out_vars;
	for (int i = 0; i < new_rank; ++i) {
		out_vars.push_back(Halide::Var());
	}

	// Build input arguments by skipping the new axis
	std::vector<Halide::Expr> in_args;
	for (int in_dim = in_shape.rank - 1; in_dim >= 0; --in_dim) {
		// Map input shape dim to output shape dim
		int out_dim = in_dim < norm_axis ? in_dim : in_dim + 1;
		int out_var_idx = new_rank - 1 - out_dim;
		in_args.push_back(out_vars[out_var_idx]);
	}

	ret(out_vars) = f(in_args);
	return ret;
}

/// @brief Infer output shape for expand_dims
inline shape_t infer_expand_dims(const shape_t& in, int axis) {
	int new_rank = in.rank + 1;
	int norm_axis = axis < 0 ? axis + new_rank : axis;

	shape_t res;
	res.rank = new_rank;
	int in_idx = 0;
	for (int i = 0; i < new_rank; ++i) {
		if (i == norm_axis) {
			res.extents[i] = 1;
		} else {
			res.extents[i] = in.extents[in_idx++];
		}
	}
	return res;
}

/// @brief Remove axes of size 1
/// @param f Input Func
/// @param in_shape Shape of input
/// @param axis Optional specific axis to squeeze (if -1, squeeze all size-1 axes)
/// @param name Function name
/// @return Func with squeezed dimensions
inline
Halide::Func squeeze(Halide::Func f, const shape_t& in_shape, int axis = -1, std::string const& name = "squeeze")
{
	std::vector<int> axes_to_keep;
	std::vector<int> axes_to_squeeze;

	if (axis == -1) {
		// Squeeze all size-1 axes
		for (int i = 0; i < in_shape.rank; ++i) {
			if (in_shape.extents[i] == 1) {
				axes_to_squeeze.push_back(i);
			} else {
				axes_to_keep.push_back(i);
			}
		}
	} else {
		int norm_axis = normalized_axis(axis, in_shape.rank);
		nh_require(nullptr, in_shape.extents[norm_axis] == 1,
			"Cannot squeeze axis %d with size %d (must be 1)",
			axis, in_shape.extents[norm_axis]);
		for (int i = 0; i < in_shape.rank; ++i) {
			if (i == norm_axis) {
				axes_to_squeeze.push_back(i);
			} else {
				axes_to_keep.push_back(i);
			}
		}
	}

	if (axes_to_squeeze.empty()) {
		// Nothing to squeeze
		Halide::Func ret(name);
		std::vector<Halide::Var> vars;
		for (int i = 0; i < in_shape.rank; ++i) {
			vars.push_back(Halide::Var());
		}
		ret(vars) = f(vars);
		return ret;
	}

	int new_rank = static_cast<int>(axes_to_keep.size());
	if (new_rank == 0) {
		// Result is scalar - return as 1D with single element
		Halide::Func ret(name);
		Halide::Var x;
		std::vector<Halide::Expr> zeros(in_shape.rank, 0);
		ret(x) = f(zeros);
		return ret;
	}

	Halide::Func ret(name);
	std::vector<Halide::Var> out_vars;
	for (int i = 0; i < new_rank; ++i) {
		out_vars.push_back(Halide::Var());
	}

	// Build input arguments
	std::vector<Halide::Expr> in_args(in_shape.rank);

	// Map output vars to input positions
	for (int out_dim = 0; out_dim < new_rank; ++out_dim) {
		int in_dim = axes_to_keep[out_dim];
		int out_var_idx = new_rank - 1 - out_dim;
		int in_arg_idx = in_shape.rank - 1 - in_dim;
		in_args[in_arg_idx] = out_vars[out_var_idx];
	}

	// Set squeezed dimensions to 0
	for (int in_dim : axes_to_squeeze) {
		int in_arg_idx = in_shape.rank - 1 - in_dim;
		in_args[in_arg_idx] = 0;
	}

	ret(out_vars) = f(in_args);
	return ret;
}

/// @brief Infer output shape for squeeze
inline shape_t infer_squeeze(const shape_t& in, int axis = -1) {
	shape_t res;
	res.rank = 0;

	if (axis == -1) {
		for (int i = 0; i < in.rank; ++i) {
			if (in.extents[i] != 1) {
				res.extents[res.rank++] = in.extents[i];
			}
		}
	} else {
		int norm_axis = normalized_axis(axis, in.rank);
		for (int i = 0; i < in.rank; ++i) {
			if (i != norm_axis) {
				res.extents[res.rank++] = in.extents[i];
			}
		}
	}

	if (res.rank == 0) {
		res.rank = 1;
		res.extents[0] = 1;
	}

	return res;
}

// -----------------------------------------------------------------------------
// Flip Operations
// -----------------------------------------------------------------------------

/// @brief Flip (reverse) array along specified axis
/// @param f Input Func
/// @param in_shape Shape of input
/// @param axis Axis to flip along
/// @param name Function name
/// @return Flipped Func
inline
Halide::Func flip(Halide::Func f, const shape_t& in_shape, int axis, std::string const& name = "flip")
{
	int norm_axis = normalized_axis(axis, in_shape.rank);
	int extent = in_shape.extents[norm_axis];

	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < in_shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	// Build input arguments, reversing the specified axis
	std::vector<Halide::Expr> in_args;
	for (int shape_dim = in_shape.rank - 1; shape_dim >= 0; --shape_dim) {
		int var_idx = in_shape.rank - 1 - shape_dim;
		if (shape_dim == norm_axis) {
			// Reverse: index becomes (extent - 1 - index)
			in_args.push_back(extent - 1 - vars[var_idx]);
		} else {
			in_args.push_back(vars[var_idx]);
		}
	}

	ret(vars) = f(in_args);
	return ret;
}

/// @brief Flip array up-down (along axis 0)
inline
Halide::Func flipud(Halide::Func f, const shape_t& in_shape, std::string const& name = "flipud")
{
	return flip(f, in_shape, 0, name);
}

/// @brief Flip array left-right (along axis 1 for 2D+)
inline
Halide::Func fliplr(Halide::Func f, const shape_t& in_shape, std::string const& name = "fliplr")
{
	nh_require(nullptr, in_shape.rank >= 2, "fliplr requires rank >= 2, got %d", in_shape.rank);
	return flip(f, in_shape, 1, name);
}

/// @brief Rotate array 90 degrees counter-clockwise k times
/// @param f Input Func (2D)
/// @param in_shape Shape of input (must be rank 2)
/// @param k Number of 90-degree rotations (default 1)
/// @param name Function name
/// @return Rotated Func
inline
Halide::Func rot90(Halide::Func f, const shape_t& in_shape, int k = 1, std::string const& name = "rot90")
{
	nh_require(nullptr, in_shape.rank == 2, "rot90 requires rank 2, got %d", in_shape.rank);

	// Normalize k to 0, 1, 2, or 3
	k = ((k % 4) + 4) % 4;

	if (k == 0) {
		// No rotation - return copy
		Halide::Func ret(name);
		Halide::Var x, y;
		ret(x, y) = f(x, y);
		return ret;
	}

	int rows = in_shape.extents[0];
	int cols = in_shape.extents[1];

	Halide::Func ret(name);
	Halide::Var x, y;

	if (k == 1) {
		// 90 degrees CCW
		// Output shape: (cols, rows)
		// new(x, y) = old(cols - 1 - y, x)
		ret(x, y) = f(cols - 1 - y, x);
	} else if (k == 2) {
		// 180 degrees
		// Output shape: same
		ret(x, y) = f(cols - 1 - x, rows - 1 - y);
	} else { // k == 3
		// 270 degrees CCW (90 CW)
		// Output shape: (cols, rows)
		// new(x, y) = old(y, rows - 1 - x)
		ret(x, y) = f(y, rows - 1 - x);
	}

	return ret;
}

/// @brief Infer output shape for rot90
inline shape_t infer_rot90(const shape_t& in, int k = 1) {
	k = ((k % 4) + 4) % 4;
	if (k == 0 || k == 2) {
		return in; // Same shape
	}
	// k == 1 or k == 3: rows and cols swap
	shape_t res = in;
	res.extents[0] = in.extents[1];
	res.extents[1] = in.extents[0];
	return res;
}

/// @brief Roll array elements along an axis
/// @param f Input Func
/// @param in_shape Shape of input
/// @param shift Number of positions to shift (positive = towards end)
/// @param axis Axis to roll along
/// @param name Function name
/// @return Rolled Func
inline
Halide::Func roll(Halide::Func f, const shape_t& in_shape, int shift, int axis, std::string const& name = "roll")
{
	int norm_axis = normalized_axis(axis, in_shape.rank);
	int extent = in_shape.extents[norm_axis];

	// Normalize shift to positive modulo
	shift = ((shift % extent) + extent) % extent;

	if (shift == 0) {
		Halide::Func ret(name);
		std::vector<Halide::Var> vars;
		for (int i = 0; i < in_shape.rank; ++i) {
			vars.push_back(Halide::Var());
		}
		ret(vars) = f(vars);
		return ret;
	}

	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < in_shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	// Build input arguments
	std::vector<Halide::Expr> in_args;
	for (int shape_dim = in_shape.rank - 1; shape_dim >= 0; --shape_dim) {
		int var_idx = in_shape.rank - 1 - shape_dim;
		if (shape_dim == norm_axis) {
			// Roll: new_index -> old_index = (new_index - shift + extent) % extent
			in_args.push_back((vars[var_idx] - shift + extent) % extent);
		} else {
			in_args.push_back(vars[var_idx]);
		}
	}

	ret(vars) = f(in_args);
	return ret;
}

/// @brief Tile (repeat) array along each axis
/// @param f Input Func
/// @param in_shape Shape of input
/// @param reps Number of repetitions for each axis
/// @param name Function name
/// @return Tiled Func
inline
Halide::Func tile(Halide::Func f, const shape_t& in_shape, const std::vector<int>& reps, std::string const& name = "tile")
{
	nh_require(nullptr, static_cast<int>(reps.size()) == in_shape.rank,
		"Tile reps length %d does not match rank %d",
		static_cast<int>(reps.size()), in_shape.rank);

	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < in_shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	// Build input arguments using modulo
	std::vector<Halide::Expr> in_args;
	for (int shape_dim = in_shape.rank - 1; shape_dim >= 0; --shape_dim) {
		int var_idx = in_shape.rank - 1 - shape_dim;
		int extent = in_shape.extents[shape_dim];
		in_args.push_back(vars[var_idx] % extent);
	}

	ret(vars) = f(in_args);
	return ret;
}

/// @brief Infer output shape for tile
inline shape_t infer_tile(const shape_t& in, const std::vector<int>& reps) {
	shape_t res = in;
	for (int i = 0; i < in.rank && i < static_cast<int>(reps.size()); ++i) {
		res.extents[i] = in.extents[i] * reps[i];
	}
	return res;
}

/// @brief Repeat elements of array along an axis
/// @param f Input Func
/// @param in_shape Shape of input
/// @param repeats Number of repetitions for each element
/// @param axis Axis along which to repeat
/// @param name Function name
/// @return Func with repeated elements
inline
Halide::Func repeat(Halide::Func f, const shape_t& in_shape, int repeats, int axis, std::string const& name = "repeat")
{
	int norm_axis = normalized_axis(axis, in_shape.rank);

	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < in_shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	// Build input arguments
	std::vector<Halide::Expr> in_args;
	for (int shape_dim = in_shape.rank - 1; shape_dim >= 0; --shape_dim) {
		int var_idx = in_shape.rank - 1 - shape_dim;
		if (shape_dim == norm_axis) {
			// Integer division to get original index
			in_args.push_back(vars[var_idx] / repeats);
		} else {
			in_args.push_back(vars[var_idx]);
		}
	}

	ret(vars) = f(in_args);
	return ret;
}

/// @brief Infer output shape for repeat
inline shape_t infer_repeat(const shape_t& in, int repeats, int axis) {
	int norm_axis = normalized_axis(axis, in.rank);
	shape_t res = in;
	res.extents[norm_axis] = in.extents[norm_axis] * repeats;
	return res;
}

/// @brief Pad mode enumeration
enum class PadMode {
	Constant,  // Pad with constant value
	Edge,      // Pad with edge values
	Reflect,   // Reflect values at boundary (not including edge)
	Symmetric  // Reflect values at boundary (including edge)
};

/// @brief Pad array with specified width and mode
/// @param f Input Func
/// @param in_shape Shape of input
/// @param pad_width Padding for each axis: {{before0, after0}, {before1, after1}, ...}
/// @param mode Padding mode
/// @param constant_value Value for constant padding
/// @param name Function name
/// @return Padded Func
inline
Halide::Func pad(Halide::Func f, const shape_t& in_shape,
                 const std::vector<std::pair<int, int>>& pad_width,
                 PadMode mode = PadMode::Constant,
                 float constant_value = 0.0f,
                 std::string const& name = "pad")
{
	nh_require(nullptr, static_cast<int>(pad_width.size()) == in_shape.rank,
		"Pad width length %d does not match rank %d",
		static_cast<int>(pad_width.size()), in_shape.rank);

	Halide::Func ret(name);
	std::vector<Halide::Var> vars;
	for (int i = 0; i < in_shape.rank; ++i) {
		vars.push_back(Halide::Var());
	}

	// Build clamped/padded input arguments
	std::vector<Halide::Expr> in_args;
	Halide::Expr in_bounds = Halide::cast<bool>(true);

	for (int shape_dim = in_shape.rank - 1; shape_dim >= 0; --shape_dim) {
		int var_idx = in_shape.rank - 1 - shape_dim;
		int extent = in_shape.extents[shape_dim];
		int pad_before = pad_width[shape_dim].first;
		int pad_after = pad_width[shape_dim].second;

		Halide::Expr idx = vars[var_idx] - pad_before;
		Halide::Expr is_in = (idx >= 0) && (idx < extent);
		in_bounds = in_bounds && is_in;

		Halide::Expr mapped_idx;
		switch (mode) {
			case PadMode::Edge:
				mapped_idx = Halide::clamp(idx, 0, extent - 1);
				break;
			case PadMode::Reflect:
				// Reflect without edge: index -1 maps to 1, -2 to 2, etc.
				mapped_idx = Halide::select(
					idx < 0, -idx,
					Halide::select(idx >= extent, 2 * extent - 2 - idx, idx)
				);
				mapped_idx = Halide::clamp(mapped_idx, 0, extent - 1);
				break;
			case PadMode::Symmetric:
				// Reflect with edge: index -1 maps to 0, -2 to 1, etc.
				mapped_idx = Halide::select(
					idx < 0, -1 - idx,
					Halide::select(idx >= extent, 2 * extent - 1 - idx, idx)
				);
				mapped_idx = Halide::clamp(mapped_idx, 0, extent - 1);
				break;
			default: // Constant
				mapped_idx = idx;
				break;
		}
		in_args.push_back(mapped_idx);
	}

	if (mode == PadMode::Constant) {
		ret(vars) = Halide::select(in_bounds, f(in_args), constant_value);
	} else {
		ret(vars) = f(in_args);
	}

	return ret;
}

/// @brief Infer output shape for pad
inline shape_t infer_pad(const shape_t& in, const std::vector<std::pair<int, int>>& pad_width) {
	shape_t res = in;
	for (int i = 0; i < in.rank && i < static_cast<int>(pad_width.size()); ++i) {
		res.extents[i] = in.extents[i] + pad_width[i].first + pad_width[i].second;
	}
	return res;
}

NS_NUM_HALIDE_END
