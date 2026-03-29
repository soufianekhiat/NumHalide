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
#include <limits>

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

    // Compute input so it can be safely read multiple times
    f.compute_root();

    Halide::Func ret(name);
    Halide::Var x("x");

    if (norm_axis == 0) {
        // Reduce along rows (axis 0 = Halide dim 1 = y), output has shape (cols,)
        Halide::RDom r(0, rows);
        ret(x) = 0;  // Initialize to row 0
        Halide::Expr best = f(x, Halide::clamp(ret(x), 0, rows - 1));
        ret(x) = Halide::select(f(x, r) < best, r, ret(x));
    } else {
        // Reduce along cols (axis 1 = Halide dim 0 = x), output has shape (rows,)
        Halide::RDom r(0, cols);
        ret(x) = 0;  // Initialize to col 0
        Halide::Expr best = f(Halide::clamp(ret(x), 0, cols - 1), x);
        ret(x) = Halide::select(f(r, x) < best, r, ret(x));
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

    // Compute input so it can be safely read multiple times
    f.compute_root();

    Halide::Func ret(name);
    Halide::Var x("x");

    if (norm_axis == 0) {
        // Reduce along rows (axis 0 = Halide dim 1 = y), output has shape (cols,)
        Halide::RDom r(0, rows);
        ret(x) = 0;  // Initialize to row 0
        Halide::Expr best = f(x, Halide::clamp(ret(x), 0, rows - 1));
        ret(x) = Halide::select(f(x, r) > best, r, ret(x));
    } else {
        // Reduce along cols (axis 1 = Halide dim 0 = x), output has shape (rows,)
        Halide::RDom r(0, cols);
        ret(x) = 0;  // Initialize to col 0
        Halide::Expr best = f(Halide::clamp(ret(x), 0, cols - 1), x);
        ret(x) = Halide::select(f(r, x) > best, r, ret(x));
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
    result.compute_root();
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

// -----------------------------------------------------------------------------
// General Sort (rank sort, O(n^2), any size)
// -----------------------------------------------------------------------------

/// @brief Sort a 1D array (ascending or descending), any size
/// @param f Input 1D Func
/// @param n Size of array
/// @param ascending If true, sort ascending; if false, descending
/// @param name Function name
/// @return Sorted 1D Func
///
/// Uses rank sort (counting sort variant): O(n^2) but works for any size.
/// For power-of-2 sizes with better performance, use bitonic_sort.
inline
Halide::Func sort_1d(Halide::Func f, int n, bool ascending = true,
    std::string const& name = "sort_1d")
{
    Halide::Var i("i");

    // Step 1: Compute rank of each element
    // rank[i] = number of elements strictly less (ascending) than f[i]
    //         + number of equal elements with smaller index (stable sort)
    Halide::Func rank_f(name + "_rank");
    rank_f(i) = Halide::cast<int32_t>(0);
    Halide::RDom rj(0, n, "rj_sort");
    Halide::Expr fi = f(i);
    Halide::Expr fj = f(rj);
    rank_f(i) += Halide::select(
        ascending
            ? (fj < fi || (fj == fi && rj < i))
            : (fj > fi || (fj == fi && rj < i)),
        Halide::cast<int32_t>(1),
        Halide::cast<int32_t>(0)
    );
    rank_f.compute_root();

    // Step 2: Gather - sorted[rank[j]] = f[j]
    Halide::Func ret(name);
    Halide::Type type = f.types()[0];
    ret(i) = Halide::Internal::make_const(type, 0);
    Halide::RDom rj2(0, n, "rj2_sort");
    ret(i) += Halide::select(
        rank_f(rj2) == i,
        f(rj2),
        Halide::Internal::make_const(type, 0)
    );

    return ret;
}

/// @brief Get indices that would sort a 1D array (argsort), any size
/// @param f Input 1D Func
/// @param n Size of array
/// @param ascending If true, sort ascending; if false, descending
/// @param name Function name
/// @return 1D Int32 Func with indices
inline
Halide::Func argsort_1d(Halide::Func f, int n, bool ascending = true,
    std::string const& name = "argsort_1d")
{
    Halide::Var i("i");

    // Compute rank (same as sort_1d)
    Halide::Func rank_f(name + "_rank");
    rank_f(i) = Halide::cast<int32_t>(0);
    Halide::RDom rj(0, n, "rj_asort");
    Halide::Expr fi = f(i);
    Halide::Expr fj = f(rj);
    rank_f(i) += Halide::select(
        ascending
            ? (fj < fi || (fj == fi && rj < i))
            : (fj > fi || (fj == fi && rj < i)),
        Halide::cast<int32_t>(1),
        Halide::cast<int32_t>(0)
    );
    rank_f.compute_root();

    // Gather indices: argsorted[rank[j]] = j
    Halide::Func ret(name);
    ret(i) = Halide::cast<int32_t>(0);
    Halide::RDom rj2(0, n, "rj2_asort");
    ret(i) += Halide::select(
        rank_f(rj2) == i,
        rj2,
        Halide::cast<int32_t>(0)
    );

    return ret;
}

// -----------------------------------------------------------------------------
// Fast Sort (O(N log^2 N) via bitonic, arbitrary N via padding)
// -----------------------------------------------------------------------------

/// @brief Sort a 1D array in O(N log^2 N) using bitonic sort with padding for non-power-of-2
inline
Halide::Func sort_1d_fast(Halide::Func f, int n, bool ascending = true,
    std::string const& name = "sort_1d_fast")
{
    int padded = 1;
    while (padded < n) padded <<= 1;

    Halide::Type type = f.types()[0];
    auto make_sentinel = [&]() -> Halide::Expr {
        if (type.is_float())
            return ascending
                ? Halide::cast(type, std::numeric_limits<float>::infinity())
                : Halide::cast(type, -std::numeric_limits<float>::infinity());
        return ascending
            ? Halide::cast(type, std::numeric_limits<int32_t>::max())
            : Halide::cast(type, std::numeric_limits<int32_t>::min());
    };

    Halide::Func work(name + "_work");
    Halide::Var x("x");
    if (padded > n)
        work(x) = Halide::select(x < n, f(Halide::clamp(x, 0, n - 1)), make_sentinel());
    else
        work(x) = f(x);
    work.compute_root();

    Halide::Func next, prev = work;
    for (int pass_size = 1; pass_size < padded; pass_size <<= 1) {
        for (int chunk_size = pass_size; chunk_size > 0; chunk_size >>= 1) {
            next = Halide::Func(name + "_s");
            Halide::Expr cs   = (x / (2 * chunk_size)) * (2 * chunk_size);
            Halide::Expr ce   = (x / (2 * chunk_size) + 1) * (2 * chunk_size);
            Halide::Expr cm   = cs + chunk_size;
            Halide::Expr partner;
            if (pass_size == chunk_size && pass_size > 1)
                partner = Halide::clamp(2 * cm - x - 1, cs, ce - 1);
            else
                partner = cs + (x - cs + chunk_size) % (2 * chunk_size);
            Halide::Expr av = prev(x), bv = prev(partner);
            if (ascending)
                next(x) = Halide::select(x < cm, Halide::min(av, bv), Halide::max(av, bv));
            else
                next(x) = Halide::select(x < cm, Halide::max(av, bv), Halide::min(av, bv));
            next.compute_root();
            prev = next;
        }
    }
    Halide::Func result(name);
    result(x) = prev(x);
    return result;
}

/// @brief Argsort a 1D array in O(N log^2 N) via bitonic sort with padding
inline
Halide::Func argsort_1d_fast(Halide::Func f, int n, bool ascending = true,
    std::string const& name = "argsort_1d_fast")
{
    int padded = 1;
    while (padded < n) padded <<= 1;

    Halide::Type type = f.types()[0];
    auto make_sentinel = [&]() -> Halide::Expr {
        if (type.is_float())
            return ascending
                ? Halide::cast(type, std::numeric_limits<float>::infinity())
                : Halide::cast(type, -std::numeric_limits<float>::infinity());
        return ascending
            ? Halide::cast(type, std::numeric_limits<int32_t>::max())
            : Halide::cast(type, std::numeric_limits<int32_t>::min());
    };

    Halide::Func indexed(name + "_idx");
    Halide::Var x("x");
    if (padded > n)
        indexed(x) = Halide::Tuple(
            Halide::select(x < n, f(Halide::clamp(x, 0, n - 1)), make_sentinel()),
            Halide::select(x < n, x, padded + n));  // sentinel index > valid range
    else
        indexed(x) = Halide::Tuple(f(x), x);
    indexed.compute_root();

    Halide::Func next, prev = indexed;
    for (int pass_size = 1; pass_size < padded; pass_size <<= 1) {
        for (int chunk_size = pass_size; chunk_size > 0; chunk_size >>= 1) {
            next = Halide::Func(name + "_s");
            Halide::Expr cs     = (x / (2 * chunk_size)) * (2 * chunk_size);
            Halide::Expr ce     = (x / (2 * chunk_size) + 1) * (2 * chunk_size);
            Halide::Expr cm     = cs + chunk_size;
            Halide::Expr partner;
            if (pass_size == chunk_size && pass_size > 1)
                partner = Halide::clamp(2 * cm - x - 1, cs, ce - 1);
            else
                partner = cs + (x - cs + chunk_size) % (2 * chunk_size);
            Halide::Expr my_v   = prev(x)[0];
            Halide::Expr my_i   = prev(x)[1];
            Halide::Expr pt_v   = prev(partner)[0];
            Halide::Expr pt_i   = prev(partner)[1];
            Halide::Expr want_me = ascending
                ? Halide::select(x < cm, my_v <= pt_v, my_v > pt_v)
                : Halide::select(x < cm, my_v >= pt_v, my_v < pt_v);
            next(x) = Halide::select(want_me,
                Halide::Tuple(my_v, my_i),
                Halide::Tuple(pt_v, pt_i));
            next.compute_root();
            prev = next;
        }
    }
    Halide::Func result(name);
    result(x) = prev(x)[1];
    return result;
}

// -----------------------------------------------------------------------------
// 2D Batched Sort via bitonic network
// -----------------------------------------------------------------------------

/// @brief Sort a 2D array along axis 1 (sort each row) or axis 0 (sort each column)
/// @param input   2D Func  f(col, row)
/// @param rows    Number of rows
/// @param cols    Number of columns
/// @param axis    1 = sort along cols (each row gets sorted), 0 = sort along rows (each col)
/// @param ascending  true = ascending order
/// @param name    Base Func name
/// @return Sorted 2D Func (same domain)
///
/// Uses the bitonic network; sort dimension is padded to next power of 2 if needed.
inline
Halide::Func sort_2d(Halide::Func input, int rows, int cols,
    int axis = 1, bool ascending = true,
    std::string const& name = "sort_2d")
{
    Halide::Var x("x"), y("y");
    Halide::Type type = input.types()[0];

    auto sentinel_expr = [&]() -> Halide::Expr {
        if (type.is_float())
            return ascending
                ? Halide::cast(type, std::numeric_limits<float>::infinity())
                : Halide::cast(type, -std::numeric_limits<float>::infinity());
        return ascending
            ? Halide::cast(type, std::numeric_limits<int32_t>::max())
            : Halide::cast(type, std::numeric_limits<int32_t>::min());
    };

    if (axis == 1) {
        // Sort along x (cols) for each fixed y (row)
        int size = cols;
        int padded = 1;
        while (padded < size) padded <<= 1;

        Halide::Func work(name + "_work");
        if (padded > size)
            work(x, y) = Halide::select(x < size, input(x, y), sentinel_expr());
        else
            work(x, y) = input(x, y);
        work.compute_root();

        Halide::Func next, prev = work;
        for (int pass_size = 1; pass_size < padded; pass_size <<= 1) {
            for (int chunk_size = pass_size; chunk_size > 0; chunk_size >>= 1) {
                next = Halide::Func(name + "_px");
                Halide::Expr cs   = (x / (2 * chunk_size)) * (2 * chunk_size);
                Halide::Expr ce   = (x / (2 * chunk_size) + 1) * (2 * chunk_size);
                Halide::Expr cm   = cs + chunk_size;
                Halide::Expr partner;
                if (pass_size == chunk_size && pass_size > 1)
                    partner = Halide::clamp(2 * cm - x - 1, cs, ce - 1);
                else
                    partner = cs + (x - cs + chunk_size) % (2 * chunk_size);
                Halide::Expr av = prev(x, y), bv = prev(partner, y);
                if (ascending)
                    next(x, y) = Halide::select(x < cm, Halide::min(av, bv), Halide::max(av, bv));
                else
                    next(x, y) = Halide::select(x < cm, Halide::max(av, bv), Halide::min(av, bv));
                next.compute_root();
                prev = next;
            }
        }
        Halide::Func result(name);
        result(x, y) = prev(x, y);
        return result;

    } else {
        // axis == 0: sort along y (rows) for each fixed x (col)
        int size = rows;
        int padded = 1;
        while (padded < size) padded <<= 1;

        Halide::Func work(name + "_work");
        if (padded > size)
            work(x, y) = Halide::select(y < size, input(x, y), sentinel_expr());
        else
            work(x, y) = input(x, y);
        work.compute_root();

        Halide::Func next, prev = work;
        for (int pass_size = 1; pass_size < padded; pass_size <<= 1) {
            for (int chunk_size = pass_size; chunk_size > 0; chunk_size >>= 1) {
                next = Halide::Func(name + "_py");
                Halide::Expr cs   = (y / (2 * chunk_size)) * (2 * chunk_size);
                Halide::Expr ce   = (y / (2 * chunk_size) + 1) * (2 * chunk_size);
                Halide::Expr cm   = cs + chunk_size;
                Halide::Expr partner;
                if (pass_size == chunk_size && pass_size > 1)
                    partner = Halide::clamp(2 * cm - y - 1, cs, ce - 1);
                else
                    partner = cs + (y - cs + chunk_size) % (2 * chunk_size);
                Halide::Expr av = prev(x, y), bv = prev(x, partner);
                if (ascending)
                    next(x, y) = Halide::select(y < cm, Halide::min(av, bv), Halide::max(av, bv));
                else
                    next(x, y) = Halide::select(y < cm, Halide::max(av, bv), Halide::min(av, bv));
                next.compute_root();
                prev = next;
            }
        }
        Halide::Func result(name);
        result(x, y) = prev(x, y);
        return result;
    }
}

/// @brief Get argsort indices for a 2D array along an axis
/// @param input  2D Func f(col, row)
/// @param rows   Number of rows
/// @param cols   Number of cols
/// @param axis   1 = argsort cols in each row, 0 = argsort rows in each col
/// @return Int32 Func with indices (same shape as input)
inline
Halide::Func argsort_2d(Halide::Func input, int rows, int cols,
    int axis = 1, bool ascending = true,
    std::string const& name = "argsort_2d")
{
    Halide::Var x("x"), y("y");
    Halide::Type type = input.types()[0];

    auto sentinel_expr = [&]() -> Halide::Expr {
        if (type.is_float())
            return ascending
                ? Halide::cast(type, std::numeric_limits<float>::infinity())
                : Halide::cast(type, -std::numeric_limits<float>::infinity());
        return ascending
            ? Halide::cast(type, std::numeric_limits<int32_t>::max())
            : Halide::cast(type, std::numeric_limits<int32_t>::min());
    };

    if (axis == 1) {
        int size = cols;
        int padded = 1;
        while (padded < size) padded <<= 1;

        Halide::Func indexed(name + "_idx");
        if (padded > size)
            indexed(x, y) = Halide::Tuple(
                Halide::select(x < size, input(x, y), sentinel_expr()),
                Halide::select(x < size, x, size + padded));
        else
            indexed(x, y) = Halide::Tuple(input(x, y), x);
        indexed.compute_root();

        Halide::Func next, prev = indexed;
        for (int pass_size = 1; pass_size < padded; pass_size <<= 1) {
            for (int chunk_size = pass_size; chunk_size > 0; chunk_size >>= 1) {
                next = Halide::Func(name + "_px");
                Halide::Expr cs   = (x / (2 * chunk_size)) * (2 * chunk_size);
                Halide::Expr ce   = (x / (2 * chunk_size) + 1) * (2 * chunk_size);
                Halide::Expr cm   = cs + chunk_size;
                Halide::Expr partner;
                if (pass_size == chunk_size && pass_size > 1)
                    partner = Halide::clamp(2 * cm - x - 1, cs, ce - 1);
                else
                    partner = cs + (x - cs + chunk_size) % (2 * chunk_size);
                Halide::Expr mv = prev(x, y)[0], mi = prev(x, y)[1];
                Halide::Expr pv = prev(partner, y)[0], pi = prev(partner, y)[1];
                Halide::Expr want_me = ascending
                    ? Halide::select(x < cm, mv <= pv, mv > pv)
                    : Halide::select(x < cm, mv >= pv, mv < pv);
                next(x, y) = Halide::select(want_me,
                    Halide::Tuple(mv, mi), Halide::Tuple(pv, pi));
                next.compute_root();
                prev = next;
            }
        }
        Halide::Func result(name);
        result(x, y) = prev(x, y)[1];
        return result;

    } else {
        int size = rows;
        int padded = 1;
        while (padded < size) padded <<= 1;

        Halide::Func indexed(name + "_idx");
        if (padded > size)
            indexed(x, y) = Halide::Tuple(
                Halide::select(y < size, input(x, y), sentinel_expr()),
                Halide::select(y < size, y, size + padded));
        else
            indexed(x, y) = Halide::Tuple(input(x, y), y);
        indexed.compute_root();

        Halide::Func next, prev = indexed;
        for (int pass_size = 1; pass_size < padded; pass_size <<= 1) {
            for (int chunk_size = pass_size; chunk_size > 0; chunk_size >>= 1) {
                next = Halide::Func(name + "_py");
                Halide::Expr cs   = (y / (2 * chunk_size)) * (2 * chunk_size);
                Halide::Expr ce   = (y / (2 * chunk_size) + 1) * (2 * chunk_size);
                Halide::Expr cm   = cs + chunk_size;
                Halide::Expr partner;
                if (pass_size == chunk_size && pass_size > 1)
                    partner = Halide::clamp(2 * cm - y - 1, cs, ce - 1);
                else
                    partner = cs + (y - cs + chunk_size) % (2 * chunk_size);
                Halide::Expr mv = prev(x, y)[0], mi = prev(x, y)[1];
                Halide::Expr pv = prev(x, partner)[0], pi = prev(x, partner)[1];
                Halide::Expr want_me = ascending
                    ? Halide::select(y < cm, mv <= pv, mv > pv)
                    : Halide::select(y < cm, mv >= pv, mv < pv);
                next(x, y) = Halide::select(want_me,
                    Halide::Tuple(mv, mi), Halide::Tuple(pv, pi));
                next.compute_root();
                prev = next;
            }
        }
        Halide::Func result(name);
        result(x, y) = prev(x, y)[1];
        return result;
    }
}

NS_NUM_HALIDE_END
