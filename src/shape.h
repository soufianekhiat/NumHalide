/// @file shape.h
/// @brief Shape engine and dimension utilities
///
/// Provides: shape_t, axis normalization, error helpers

#pragma once

#include "common.h"
#include <algorithm>
#include <string>
#include <vector>
#include <sstream>
#include <initializer_list>

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Error Handling
// -----------------------------------------------------------------------------

#include <cstdio>
#include <stdexcept>

/// @brief Require condition to be true, otherwise throw std::runtime_error.
/// @param cond Condition to check
/// @param fmt  printf-style format string for the error message
/// @param ...  Format arguments
#define nh_require(cond, fmt, ...) \
	do { \
		if(!(cond)) { \
			char _nh_buf[1024]; \
			snprintf(_nh_buf, 1024, fmt, ##__VA_ARGS__); \
			throw std::runtime_error(_nh_buf); \
		} \
	} while(0)

// -----------------------------------------------------------------------------
// Shape Definition
// -----------------------------------------------------------------------------

/// @brief Fixed-rank shape descriptor (up to 8 dimensions)
struct shape_t {
	static constexpr int MAX_RANK = 8;
	int rank = 0;
	int extents[MAX_RANK] = { 1, 1, 1, 1, 1, 1, 1, 1 };

	shape_t() = default;

	shape_t(std::initializer_list<int> dims) {
		rank = static_cast<int>(dims.size());
		nh_require(rank <= MAX_RANK,
			"shape_t: rank %d exceeds MAX_RANK (%d)", rank, MAX_RANK);
		int i = 0;
		for (int d : dims) {
			extents[i++] = d;
		}
	}

	shape_t(const std::vector<int>& dims) {
		rank = static_cast<int>(dims.size());
		nh_require(rank <= MAX_RANK,
			"shape_t: rank %d exceeds MAX_RANK (%d)", rank, MAX_RANK);
		for (int i = 0; i < rank; ++i) {
			extents[i] = dims[i];
		}
	}

	int operator[](int i) const {
		nh_require(i >= 0 && i < rank,
			"shape_t: index %d out of bounds [0, %d)", i, rank);
		return extents[i];
	}

	int& operator[](int i) {
		nh_require(i >= 0 && i < rank,
			"shape_t: index %d out of bounds [0, %d)", i, rank);
		return extents[i];
	}

	bool operator==(const shape_t& other) const {
		if (rank != other.rank) return false;
		for (int i = 0; i < rank; ++i) {
			if (extents[i] != other.extents[i]) return false;
		}
		return true;
	}

	bool operator!=(const shape_t& other) const {
		return !(*this == other);
	}
};

/// @brief Convert shape to string for debugging
inline std::string shape_to_string(const shape_t& s) {
	std::stringstream ss;
	ss << "(";
	for (int i = 0; i < s.rank; ++i) {
		ss << s.extents[i];
		if (i < s.rank - 1) ss << ", ";
	}
	ss << ")";
	return ss.str();
}

// -----------------------------------------------------------------------------
// Axis Utilities
// -----------------------------------------------------------------------------

/// @brief Normalize axis index (handle negative indices)
/// @param axis Input axis
/// @param rank Tensor rank
/// @return Normalized axis in [0, rank)
inline int normalized_axis(int axis, int rank) {
	if (axis < 0) axis += rank;
	nh_require(axis >= 0 && axis < rank,
		"Axis %d out of bounds for rank-%d tensor", axis, rank);
	return axis;
}

/// @brief Check if two shapes are identical except at a specific axis
inline bool check_same_except(const shape_t& a, const shape_t& b, int axis) {
	if (a.rank != b.rank) return false;
	for (int i = 0; i < a.rank; ++i) {
		if (i != axis && a.extents[i] != b.extents[i]) return false;
	}
	return true;
}

/// @brief Infer output shape for concatenation
inline shape_t infer_concat(const shape_t& a, const shape_t& b, int axis) {
	int norm_axis = normalized_axis(axis, a.rank);
	nh_require(check_same_except(a, b, norm_axis), 
		"Shapes %s and %s mismatch for concat at axis %d", 
		shape_to_string(a).c_str(), shape_to_string(b).c_str(), axis);
	
	shape_t res = a;
	res.extents[norm_axis] += b.extents[norm_axis];
	return res;
}

/// @brief Infer output shape for reduction
inline shape_t infer_reduce(const shape_t& in, const std::vector<int>& axes, bool keepdims) {
	shape_t res;
	if (keepdims) {
		res = in;
		for (int axis : axes) {
			int norm_axis = normalized_axis(axis, in.rank);
			res.extents[norm_axis] = 1;
		}
	} else {
		// Remove reduced dimensions
		res.rank = 0;
		for (int i = 0; i < in.rank; ++i) {
			bool reduced = false;
			for (int axis : axes) {
				if (normalized_axis(axis, in.rank) == i) {
					reduced = true;
					break;
				}
			}
			if (!reduced) {
				res.extents[res.rank++] = in.extents[i];
			}
		}
	}
	return res;
}

/// @brief Infer output shape for broadcasting
inline shape_t infer_broadcast(const shape_t& a, const shape_t& b) {
	shape_t res;
	res.rank = std::max(a.rank, b.rank);

	for (int i = 0; i < res.rank; ++i) {
		// Align dimensions to the right
		int a_idx = a.rank - 1 - i;
		int b_idx = b.rank - 1 - i;
		int res_idx = res.rank - 1 - i;

		int dim_a = (a_idx >= 0) ? a.extents[a_idx] : 1;
		int dim_b = (b_idx >= 0) ? b.extents[b_idx] : 1;

		if (dim_a == dim_b) {
			res.extents[res_idx] = dim_a;
		} else if (dim_a == 1) {
			res.extents[res_idx] = dim_b;
		} else if (dim_b == 1) {
			res.extents[res_idx] = dim_a;
		} else {
			nh_require(false,
				"Operands could not be broadcast together with shapes %s and %s",
				shape_to_string(a).c_str(), shape_to_string(b).c_str());
		}
	}
	return res;
}

/// @brief Infer output shape for slice operation
/// @param in Input shape
/// @param axis Axis to slice along
/// @param start Start index
/// @param stop Stop index (exclusive)
/// @param step Step size
/// @return Output shape with reduced dimension at axis
inline shape_t infer_slice(const shape_t& in, int axis, int start, int stop, int step) {
	int norm_axis = normalized_axis(axis, in.rank);
	int extent = in.extents[norm_axis];

	// Normalize negative indices
	if (start < 0) start += extent;
	if (stop < 0) stop += extent;

	// Clamp to valid range
	start = std::max(0, std::min(start, extent));
	stop = std::max(0, std::min(stop, extent));

	nh_require(step != 0, "Slice step cannot be zero");

	// Compute output size
	int out_size = 0;
	if (step > 0 && stop > start) {
		out_size = (stop - start + step - 1) / step;
	} else if (step < 0 && start > stop) {
		out_size = (start - stop - step - 1) / (-step);
	}

	shape_t res = in;
	res.extents[norm_axis] = out_size;
	return res;
}

/// @brief Infer output shape for transpose operation
/// @param in Input shape
/// @param axes Permutation of dimensions
/// @return Transposed shape
inline shape_t infer_transpose(const shape_t& in, const std::vector<int>& axes) {
	nh_require(static_cast<int>(axes.size()) == in.rank,
		"Transpose axes length %d does not match rank %d",
		static_cast<int>(axes.size()), in.rank);

	// Check for valid permutation
	std::vector<bool> seen(in.rank, false);
	for (int ax : axes) {
		int norm = normalized_axis(ax, in.rank);
		nh_require(!seen[norm], "Duplicate axis %d in transpose", ax);
		seen[norm] = true;
	}

	shape_t res;
	res.rank = in.rank;
	for (int i = 0; i < in.rank; ++i) {
		int src_axis = normalized_axis(axes[i], in.rank);
		res.extents[i] = in.extents[src_axis];
	}
	return res;
}

/// @brief Compute axes permutation for moveaxis
/// @param rank Tensor rank
/// @param src Source axis position
/// @param dst Destination axis position
/// @return Permutation array for transpose
inline std::vector<int> compute_moveaxis_perm(int rank, int src, int dst) {
	int norm_src = normalized_axis(src, rank);
	int norm_dst = normalized_axis(dst, rank);

	std::vector<int> axes;
	for (int i = 0; i < rank; ++i) {
		axes.push_back(i);
	}

	// Remove src and insert at dst
	axes.erase(axes.begin() + norm_src);
	axes.insert(axes.begin() + norm_dst, norm_src);

	// Now axes[i] tells us where output dimension i came from
	// But we need the inverse: for each output dim, which input dim maps there
	std::vector<int> result(rank);
	for (int i = 0; i < rank; ++i) {
		result[i] = axes[i];
	}
	return result;
}

NS_NUM_HALIDE_END
