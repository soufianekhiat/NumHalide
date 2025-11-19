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

/// @brief Require condition to be true, otherwise report error
/// @param user_context Halide user context (can be nullptr)
/// @param cond Condition to check
/// @param fmt Format string
/// @param ... Arguments
#define nh_require(user_context, cond, fmt, ...) \
	do { \
		if(!(cond)) { \
			char buf[1024]; \
			snprintf(buf, 1024, fmt, ##__VA_ARGS__); \
			if (user_context) { \
				/*halide_error(user_context, buf);*/ \
				throw std::runtime_error(buf); \
			} else { \
				throw std::runtime_error(buf); \
			} \
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
		NH_ASSERT(rank <= MAX_RANK && "Rank exceeds maximum supported (8)");
		int i = 0;
		for (int d : dims) {
			extents[i++] = d;
		}
	}

	shape_t(const std::vector<int>& dims) {
		rank = static_cast<int>(dims.size());
		NH_ASSERT(rank <= MAX_RANK && "Rank exceeds maximum supported (8)");
		for (int i = 0; i < rank; ++i) {
			extents[i] = dims[i];
		}
	}

	int operator[](int i) const {
		NH_ASSERT(i >= 0 && i < rank);
		return extents[i];
	}

	int& operator[](int i) {
		NH_ASSERT(i >= 0 && i < rank);
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
	NH_ASSERT(axis >= 0 && axis < rank && "Axis out of bounds");
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
	nh_require(nullptr, check_same_except(a, b, norm_axis), 
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
			nh_require(nullptr, false, 
				"Operands could not be broadcast together with shapes %s and %s",
				shape_to_string(a).c_str(), shape_to_string(b).c_str());
		}
	}
	return res;
}

NS_NUM_HALIDE_END
