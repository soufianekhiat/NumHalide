/// @file sort.h
/// @brief Sorting and search operations
///
/// Provides: argmin, argmax, searchsorted
///
/// Note: Full sorting (sort, argsort) requires compile-time known sizes
/// and is complex to implement efficiently in Halide. See Halide's
/// bitonic_sort in test/performance/sort.cpp for reference.

#pragma once

#include "common.h"
#include "numhalide.h"
#include "shape.h"
#include "reduce.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Index of Extremum
// -----------------------------------------------------------------------------

/// @brief Find the index of the minimum value (1D array)
/// @param f Input Func (1D)
/// @param in_shape Shape of input
/// @param name Function name
/// @return Func returning scalar index of minimum
inline
Halide::Func argmin(Halide::Func f, const shape_t& in_shape, std::string const& name = "argmin")
{
    nh_require(nullptr, in_shape.rank == 1, "argmin currently supports 1D arrays only");

    int n = in_shape.extents[0];

    Halide::Func ret(name);
    Halide::RDom r(0, n);

    // Use argmin to find index of minimum
    Halide::Expr val = f(r);
    Halide::Tuple result = Halide::argmin(r, val);

    Halide::Var idx;
    ret(idx) = result[0];  // The index

    return ret;
}

/// @brief Find the index of the minimum value along an axis (2D only)
/// @param f Input Func
/// @param in_shape Shape of input (must be 2D)
/// @param axis Axis to reduce along
/// @param name Function name
/// @return Func with indices of minima
inline
Halide::Func argmin(Halide::Func f, const shape_t& in_shape, int axis, std::string const& name = "argmin")
{
    nh_require(nullptr, in_shape.rank == 2, "argmin with axis currently supports 2D only");

    int norm_axis = normalized_axis(axis, in_shape.rank);
    int rows = in_shape.extents[0];
    int cols = in_shape.extents[1];

    Halide::Func ret(name);
    Halide::Var x("x");

    if (norm_axis == 0) {
        // Reduce along rows (axis 0), output has shape (cols,)
        Halide::RDom r(0, rows);
        Halide::Tuple result = Halide::argmin(r, f(x, r));
        ret(x) = result[0];
    } else {
        // Reduce along cols (axis 1), output has shape (rows,)
        Halide::RDom r(0, cols);
        Halide::Tuple result = Halide::argmin(r, f(r, x));
        ret(x) = result[0];
    }

    return ret;
}

/// @brief Find the index of the maximum value (1D array)
/// @param f Input Func (1D)
/// @param in_shape Shape of input
/// @param name Function name
/// @return Func returning scalar index of maximum
inline
Halide::Func argmax(Halide::Func f, const shape_t& in_shape, std::string const& name = "argmax")
{
    nh_require(nullptr, in_shape.rank == 1, "argmax currently supports 1D arrays only");

    int n = in_shape.extents[0];

    Halide::Func ret(name);
    Halide::RDom r(0, n);

    // Use argmax to find index of maximum
    Halide::Expr val = f(r);
    Halide::Tuple result = Halide::argmax(r, val);

    Halide::Var idx;
    ret(idx) = result[0];

    return ret;
}

/// @brief Find the index of the maximum value along an axis (2D only)
/// @param f Input Func
/// @param in_shape Shape of input (must be 2D)
/// @param axis Axis to reduce along
/// @param name Function name
/// @return Func with indices of maxima
inline
Halide::Func argmax(Halide::Func f, const shape_t& in_shape, int axis, std::string const& name = "argmax")
{
    nh_require(nullptr, in_shape.rank == 2, "argmax with axis currently supports 2D only");

    int norm_axis = normalized_axis(axis, in_shape.rank);
    int rows = in_shape.extents[0];
    int cols = in_shape.extents[1];

    Halide::Func ret(name);
    Halide::Var x("x");

    if (norm_axis == 0) {
        // Reduce along rows (axis 0), output has shape (cols,)
        Halide::RDom r(0, rows);
        Halide::Tuple result = Halide::argmax(r, f(x, r));
        ret(x) = result[0];
    } else {
        // Reduce along cols (axis 1), output has shape (rows,)
        Halide::RDom r(0, cols);
        Halide::Tuple result = Halide::argmax(r, f(r, x));
        ret(x) = result[0];
    }

    return ret;
}

// -----------------------------------------------------------------------------
// Binary Search
// -----------------------------------------------------------------------------

/// @brief Find insertion indices for values in a sorted array (1D)
/// @param sorted_array Sorted 1D Func
/// @param values Values to search for
/// @param array_size Size of sorted array
/// @param values_size Number of values to search
/// @param name Function name
/// @return Func with insertion indices
///
/// Returns indices where values should be inserted to maintain sort order.
/// This is equivalent to numpy.searchsorted with side='left'.
inline
Halide::Func searchsorted(Halide::Func sorted_array, Halide::Func values,
                          int array_size, int values_size,
                          std::string const& name = "searchsorted")
{
    Halide::Func ret(name);
    Halide::Var i;

    // Binary search using reduction
    // We'll iterate log2(array_size) times
    int max_iters = 0;
    int temp = array_size;
    while (temp > 0) {
        max_iters++;
        temp >>= 1;
    }

    // For each value, find insertion point
    Halide::RDom r(0, max_iters);

    // State: (low, high)
    Halide::Func state("state");
    state(i) = Halide::Tuple(0, array_size);

    Halide::Func binary_search("binary_search");
    binary_search(i, r) = Halide::Tuple(
        Halide::undef<int32_t>(),
        Halide::undef<int32_t>()
    );

    // Initialize
    binary_search(i, -1) = Halide::Tuple(0, array_size);

    // Binary search iteration
    Halide::Expr low = binary_search(i, r - 1)[0];
    Halide::Expr high = binary_search(i, r - 1)[1];
    Halide::Expr mid = (low + high) / 2;
    Halide::Expr val = values(i);
    Halide::Expr arr_val = sorted_array(Halide::clamp(mid, 0, array_size - 1));

    binary_search(i, r) = Halide::select(
        low >= high,
        Halide::Tuple(low, high),
        Halide::select(
            arr_val < val,
            Halide::Tuple(mid + 1, high),
            Halide::Tuple(low, mid)
        )
    );

    ret(i) = binary_search(i, max_iters - 1)[0];

    return ret;
}

/// @brief Simple searchsorted for a single value
inline
Halide::Func searchsorted_single(Halide::Func sorted_array, Halide::Expr value,
                                  int array_size, std::string const& name = "searchsorted")
{
    Halide::Func values("search_value");
    Halide::Var i;
    values(i) = value;

    return searchsorted(sorted_array, values, array_size, 1, name);
}

// -----------------------------------------------------------------------------
// Bitonic Sort (compile-time size, power of 2)
// -----------------------------------------------------------------------------

/// @brief Sort a 1D array using bitonic sort
/// @param input Input 1D Func
/// @param size Size of array (must be power of 2)
/// @param name Function name
/// @return Sorted Func
///
/// Note: Size must be a power of 2 and known at compile time.
/// For arbitrary sizes, pad to next power of 2.
inline
Halide::Func bitonic_sort(Halide::Func input, int size, std::string const& name = "sorted")
{
    // Verify size is power of 2
    nh_require(nullptr, (size & (size - 1)) == 0, "bitonic_sort requires power of 2 size, got %d", size);

    Halide::Func next, prev = input;
    Halide::Var x("x");

    for (int pass_size = 1; pass_size < size; pass_size <<= 1) {
        for (int chunk_size = pass_size; chunk_size > 0; chunk_size >>= 1) {
            next = Halide::Func(name + "_pass");

            Halide::Expr chunk_start = (x / (2 * chunk_size)) * (2 * chunk_size);
            Halide::Expr chunk_end = (x / (2 * chunk_size) + 1) * (2 * chunk_size);
            Halide::Expr chunk_middle = chunk_start + chunk_size;

            if (pass_size == chunk_size && pass_size > 1) {
                // Flipped pass
                Halide::Expr partner = 2 * chunk_middle - x - 1;
                partner = Halide::clamp(partner, chunk_start, chunk_end - 1);
                next(x) = Halide::select(x < chunk_middle,
                                 Halide::min(prev(x), prev(partner)),
                                 Halide::max(prev(x), prev(partner)));
            } else {
                // Regular pass
                Halide::Expr chunk_index = x - chunk_start;
                Halide::Expr partner = chunk_start + (chunk_index + chunk_size) % (chunk_size * 2);
                next(x) = Halide::select(x < chunk_middle,
                                 Halide::min(prev(x), prev(partner)),
                                 Halide::max(prev(x), prev(partner)));
            }

            next.compute_root();
            prev = next;
        }
    }

    // Rename final result
    Halide::Func result(name);
    result(x) = next(x);
    return result;
}

/// @brief Get argsort indices using bitonic sort
/// @param input Input 1D Func
/// @param size Size of array (must be power of 2)
/// @param name Function name
/// @return Func returning indices that would sort the array
inline
Halide::Func bitonic_argsort(Halide::Func input, int size, std::string const& name = "argsort")
{
    nh_require(nullptr, (size & (size - 1)) == 0, "bitonic_argsort requires power of 2 size, got %d", size);

    // Create tuples of (value, original_index)
    Halide::Func indexed("indexed");
    Halide::Var x("x");
    indexed(x) = Halide::Tuple(input(x), x);

    Halide::Func next, prev = indexed;

    for (int pass_size = 1; pass_size < size; pass_size <<= 1) {
        for (int chunk_size = pass_size; chunk_size > 0; chunk_size >>= 1) {
            next = Halide::Func(name + "_pass");

            Halide::Expr chunk_start = (x / (2 * chunk_size)) * (2 * chunk_size);
            Halide::Expr chunk_end = (x / (2 * chunk_size) + 1) * (2 * chunk_size);
            Halide::Expr chunk_middle = chunk_start + chunk_size;

            Halide::Expr partner;
            if (pass_size == chunk_size && pass_size > 1) {
                partner = 2 * chunk_middle - x - 1;
                partner = Halide::clamp(partner, chunk_start, chunk_end - 1);
            } else {
                Halide::Expr chunk_index = x - chunk_start;
                partner = chunk_start + (chunk_index + chunk_size) % (chunk_size * 2);
            }

            Halide::Expr my_val = prev(x)[0];
            Halide::Expr my_idx = prev(x)[1];
            Halide::Expr partner_val = prev(partner)[0];
            Halide::Expr partner_idx = prev(partner)[1];

            Halide::Expr take_smaller = x < chunk_middle;
            Halide::Expr is_smaller = my_val < partner_val;

            next(x) = Halide::select(
                (take_smaller && is_smaller) || (!take_smaller && !is_smaller),
                Halide::Tuple(my_val, my_idx),
                Halide::Tuple(partner_val, partner_idx)
            );

            next.compute_root();
            prev = next;
        }
    }

    // Extract indices
    Halide::Func result(name);
    result(x) = next(x)[1];
    return result;
}

NS_NUM_HALIDE_END
