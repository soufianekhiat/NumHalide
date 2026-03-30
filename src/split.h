/// @file split.h
/// @brief Array splitting operations
///
/// Provides: split, split_at, hsplit, vsplit

#pragma once

#include "common.h"
#include "shape.h"
#include <vector>
#include <string>

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Split into N Equal Sections
// -----------------------------------------------------------------------------

/// @brief Split an array into N equal sections along an axis
/// @param f Input Func
/// @param shape Shape of input
/// @param axis Axis to split along
/// @param n_sections Number of equal sections
/// @param name Base name for output Funcs
/// @return Vector of Funcs, each representing one section
///
/// The extent along the split axis must be evenly divisible by n_sections.
/// Each output Func has shape.extents[axis] / n_sections elements along that axis.
///
/// Usage:
///   auto parts = split(f, {12}, 0, 3);  // three 1D funcs of size 4
///   auto parts = split(f, {6, 8}, 1, 4); // four 6x2 funcs
inline
std::vector<Halide::Func> split(Halide::Func f, const shape_t& shape, int axis, int n_sections,
                                std::string const& name = "split")
{
	int norm_axis = normalized_axis(axis, shape.rank);
	int extent = shape.extents[norm_axis];

	nh_require(n_sections > 0, "split: n_sections must be > 0");
	nh_require(extent % n_sections == 0,
		"split: extent %d along axis %d is not evenly divisible by %d sections",
		extent, axis, n_sections);

	int section_size = extent / n_sections;

	std::vector<Halide::Func> result;
	for (int i = 0; i < n_sections; ++i) {
		Halide::Func part(name + "_" + std::to_string(i));

		if (shape.rank == 1) {
			Halide::Var x;
			part(x) = f(x + i * section_size);
		}
		else if (shape.rank == 2) {
			Halide::Var x, y;
			if (norm_axis == 0) {
				// Split along rows (axis 0 = Halide dim 1 = y)
				part(x, y) = f(x, y + i * section_size);
			}
			else {
				// Split along cols (axis 1 = Halide dim 0 = x)
				part(x, y) = f(x + i * section_size, y);
			}
		}
		else if (shape.rank == 3) {
			Halide::Var x, y, z;
			if (norm_axis == 0) {
				part(x, y, z) = f(x, y, z + i * section_size);
			}
			else if (norm_axis == 1) {
				part(x, y, z) = f(x, y + i * section_size, z);
			}
			else {
				part(x, y, z) = f(x + i * section_size, y, z);
			}
		}
		else {
			nh_require(false,
				"split: rank %d not yet supported (max 3D)", shape.rank);
		}

		result.push_back(part);
	}

	return result;
}

// -----------------------------------------------------------------------------
// Split at Given Indices
// -----------------------------------------------------------------------------

/// @brief Split an array at specified indices along an axis
/// @param f Input Func
/// @param shape Shape of input
/// @param axis Axis to split along
/// @param indices Indices at which to split (sorted, exclusive boundaries)
/// @param name Base name for output Funcs
/// @return Vector of Funcs, one more than the number of indices
///
/// For a 1D array of size 10 split at {3, 7}:
///   part 0: elements [0, 3)  -> f(x) for x in [0,3)
///   part 1: elements [3, 7)  -> f(x+3) for x in [0,4)
///   part 2: elements [7, 10) -> f(x+7) for x in [0,3)
///
/// Usage:
///   auto parts = split_at(f, {10}, 0, {3, 7});
inline
std::vector<Halide::Func> split_at(Halide::Func f, const shape_t& shape, int axis,
                                    const std::vector<int>& indices,
                                    std::string const& name = "split_at")
{
	nh_require(shape.rank == 1, "split_at currently supports 1D arrays only");
	int norm_axis = normalized_axis(axis, shape.rank);
	int extent = shape.extents[norm_axis];

	// Build boundary list: [0, indices..., extent]
	std::vector<int> boundaries;
	boundaries.push_back(0);
	for (int idx : indices) {
		nh_require(idx >= 0 && idx <= extent,
			"split_at: index %d out of range [0, %d]", idx, extent);
		boundaries.push_back(idx);
	}
	boundaries.push_back(extent);

	std::vector<Halide::Func> result;
	for (int i = 0; i + 1 < static_cast<int>(boundaries.size()); ++i) {
		int start = boundaries[i];
		Halide::Func part(name + "_" + std::to_string(i));
		Halide::Var x;
		part(x) = f(x + start);
		result.push_back(part);
	}

	return result;
}

// -----------------------------------------------------------------------------
// Convenience: hsplit / vsplit
// -----------------------------------------------------------------------------

/// @brief Split along axis 1 (columns for 2D arrays)
/// @param f Input Func
/// @param shape Shape of input (must be >= 2D)
/// @param n_sections Number of equal sections
/// @param name Base name for output Funcs
/// @return Vector of Funcs
///
/// Equivalent to split(f, shape, 1, n_sections).
/// For a 2D array, splits along columns.
///
/// Usage:
///   auto parts = hsplit(f, {4, 8}, 4);  // four 4x2 funcs
inline
std::vector<Halide::Func> hsplit(Halide::Func f, const shape_t& shape, int n_sections,
                                 std::string const& name = "hsplit")
{
	nh_require(shape.rank >= 2,
		"hsplit: input must be at least 2D, got rank %d", shape.rank);
	return split(f, shape, 1, n_sections, name);
}

/// @brief Split along axis 0 (rows for 2D arrays)
/// @param f Input Func
/// @param shape Shape of input
/// @param n_sections Number of equal sections
/// @param name Base name for output Funcs
/// @return Vector of Funcs
///
/// Equivalent to split(f, shape, 0, n_sections).
/// For a 2D array, splits along rows.
///
/// Usage:
///   auto parts = vsplit(f, {8, 4}, 4);  // four 2x4 funcs
inline
std::vector<Halide::Func> vsplit(Halide::Func f, const shape_t& shape, int n_sections,
                                 std::string const& name = "vsplit")
{
	return split(f, shape, 0, n_sections, name);
}

NS_NUM_HALIDE_END
