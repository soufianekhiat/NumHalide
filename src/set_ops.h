/// @file set_ops.h
/// @brief Set operations for arrays
///
/// Provides: unique, in1d, intersect1d, union1d, setdiff1d
///
/// Note: Set operations require sorting. For efficiency, these operations
/// work on arrays with sizes that are powers of 2 (using bitonic sort).

#pragma once

#include "common.h"
#include "numhalide.h"
#include "shape.h"
#include "sort.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Unique Elements
// -----------------------------------------------------------------------------

/// @brief Mark unique elements in a sorted array
/// @param sorted_input Sorted 1D Func
/// @param size Size of array
/// @param name Function name
/// @return Tuple Func: (value, is_first_occurrence) where is_first_occurrence is 1 for unique elements
///
/// Usage: Sort first with bitonic_sort, then use this to identify unique elements.
/// The output marks the first occurrence of each value with 1, duplicates with 0.
inline
Halide::Func mark_unique(Halide::Func sorted_input, int size, std::string const& name = "mark_unique")
{
    Halide::Func ret(name);
    Halide::Var x("x");

    // First element is always unique, others are unique if different from previous
    Halide::Expr is_first = (x == 0);
    Halide::Expr prev_val = sorted_input(Halide::max(x - 1, 0));
    Halide::Expr curr_val = sorted_input(x);
    Halide::Expr is_different = (curr_val != prev_val);

    ret(x) = Halide::Tuple(curr_val, Halide::select(is_first || is_different, 1, 0));

    return ret;
}

/// @brief Count unique elements in a sorted array
/// @param sorted_input Sorted 1D Func
/// @param size Size of array (must be power of 2)
/// @param name Function name
/// @return Func returning count of unique elements
inline
Halide::Func count_unique(Halide::Func sorted_input, int size, std::string const& name = "count_unique")
{
    Halide::Func marked = mark_unique(sorted_input, size, name + "_marked");

    Halide::Func ret(name);
    Halide::Var x("x");
    Halide::RDom r(0, size);

    ret(x) = Halide::sum(marked(r)[1]);

    return ret;
}

/// @brief Get unique elements from a 1D array
/// @param input Input 1D Func
/// @param size Size of array (must be power of 2)
/// @param name Function name
/// @return Sorted array with unique elements (padded with last value)
///
/// Note: Since Halide Funcs must have fixed output size, the result
/// is padded. Use count_unique() to get the actual number of unique elements.
inline
Halide::Func unique(Halide::Func input, int size, std::string const& name = "unique")
{
    nh_require(nullptr, (size & (size - 1)) == 0, "unique requires power of 2 size, got %d", size);

    // First, sort the input
    auto sorted = bitonic_sort(input, size, name + "_sorted");

    // Mark unique elements
    auto marked = mark_unique(sorted, size, name + "_marked");

    // Compute prefix sum of marks to get destination indices
    // This is a parallel prefix sum (scan)
    Halide::Func prefix(name + "_prefix");
    Halide::Var x("x");

    // Simple sequential prefix sum (for small arrays)
    // For larger arrays, a parallel scan would be more efficient
    Halide::RDom r(0, size);
    prefix(x) = 0;
    prefix(r) = prefix(Halide::max(r - 1, 0)) + marked(Halide::max(r - 1, 0))[1];

    // Scatter unique values to their compacted positions
    // Since Halide doesn't support scatter directly, we use a gather approach
    Halide::Func ret(name);

    // For each output position, find the source position
    // This is expensive but works for small arrays
    Halide::RDom scan(0, size);
    Halide::Func find_src(name + "_find_src");
    find_src(x) = size - 1;  // Default to last element
    find_src(x) = Halide::select(
        prefix(scan) == x && marked(scan)[1] == 1,
        scan,
        find_src(x)
    );

    ret(x) = sorted(find_src(x));

    return ret;
}

// -----------------------------------------------------------------------------
// Membership Test
// -----------------------------------------------------------------------------

/// @brief Test if elements are in a sorted array (linear search for now)
/// @param test_values Values to test
/// @param sorted_array Sorted array to search in
/// @param test_size Number of values to test
/// @param array_size Size of sorted array
/// @param name Function name
/// @return Func returning 1 if element is found, 0 otherwise
///
/// Note: Uses linear search. For large arrays, consider using binary search.
inline
Halide::Func in1d(Halide::Func test_values, Halide::Func sorted_array,
                  int test_size, int array_size,
                  std::string const& name = "in1d")
{
    Halide::Func ret(name);
    Halide::Var i("i");
    Halide::RDom r(0, array_size);

    // For each test value, check if it exists anywhere in the sorted array
    Halide::Expr val = test_values(i);

    // Use maximum to check if any element matches
    ret(i) = Halide::maximum(Halide::select(sorted_array(r) == val, 1, 0));
    ret.compute_root();

    return ret;
}

// -----------------------------------------------------------------------------
// Set Operations
// -----------------------------------------------------------------------------

/// @brief Compute intersection of two sorted arrays
/// @param sorted_a First sorted array
/// @param sorted_b Second sorted array
/// @param size_a Size of first array
/// @param size_b Size of second array
/// @param name Function name
/// @return Tuple: (intersection values, valid mask)
///
/// Note: Returns array of size min(size_a, size_b) with valid elements marked.
inline
Halide::Func intersect1d_sorted(Halide::Func sorted_a, Halide::Func sorted_b,
                                 int size_a, int size_b,
                                 std::string const& name = "intersect1d")
{
    // For each element in a, check if it exists in b
    auto in_b = in1d(sorted_a, sorted_b, size_a, size_b, name + "_in_b");

    // Also check if it's the first occurrence (unique in a)
    auto marked_a = mark_unique(sorted_a, size_a, name + "_marked_a");
    marked_a.compute_root();

    Halide::Func ret(name);
    Halide::Var x("x");

    // Element is in intersection if: it's unique in a AND exists in b
    Halide::Expr is_valid = (marked_a(x)[1] == 1) && (in_b(x) == 1);

    ret(x) = Halide::Tuple(sorted_a(x), Halide::select(is_valid, 1, 0));

    return ret;
}

/// @brief Compute set difference (elements in a not in b) for sorted arrays
/// @param sorted_a First sorted array
/// @param sorted_b Second sorted array
/// @param size_a Size of first array
/// @param size_b Size of second array
/// @param name Function name
/// @return Tuple: (difference values, valid mask)
inline
Halide::Func setdiff1d_sorted(Halide::Func sorted_a, Halide::Func sorted_b,
                               int size_a, int size_b,
                               std::string const& name = "setdiff1d")
{
    // For each element in a, check if it exists in b
    auto in_b = in1d(sorted_a, sorted_b, size_a, size_b, name + "_in_b");

    // Also check if it's the first occurrence (unique in a)
    auto marked_a = mark_unique(sorted_a, size_a, name + "_marked_a");
    marked_a.compute_root();

    Halide::Func ret(name);
    Halide::Var x("x");

    // Element is in difference if: it's unique in a AND NOT exists in b
    Halide::Expr is_valid = (marked_a(x)[1] == 1) && (in_b(x) == 0);

    ret(x) = Halide::Tuple(sorted_a(x), Halide::select(is_valid, 1, 0));

    return ret;
}

/// @brief Compute union of two sorted arrays
/// @param sorted_a First sorted array
/// @param sorted_b Second sorted array
/// @param size_a Size of first array
/// @param size_b Size of second array
/// @param name Function name
/// @return Tuple: (union values, valid mask) with size = size_a + size_b
///
/// Note: The result is the concatenation of unique(a) and setdiff(b, a).
/// Use the valid mask to identify actual union elements.
inline
Halide::Func union1d_sorted(Halide::Func sorted_a, Halide::Func sorted_b,
                             int size_a, int size_b,
                             std::string const& name = "union1d")
{
    // Get unique elements from a
    auto marked_a = mark_unique(sorted_a, size_a, name + "_marked_a");
    marked_a.compute_root();

    // Get elements in b that are not in a
    auto in_a = in1d(sorted_b, sorted_a, size_b, size_a, name + "_in_a");
    auto marked_b = mark_unique(sorted_b, size_b, name + "_marked_b");
    marked_b.compute_root();

    Halide::Func ret(name);
    Halide::Var x("x");

    // First size_a elements: unique elements from a
    // Next size_b elements: elements from b not in a (and unique in b)
    Halide::Expr in_a_section = x < size_a;

    Halide::Expr val = Halide::select(
        in_a_section,
        sorted_a(x),
        sorted_b(x - size_a)
    );

    Halide::Expr is_valid = Halide::select(
        in_a_section,
        marked_a(x)[1] == 1,
        (marked_b(x - size_a)[1] == 1) && (in_a(x - size_a) == 0)
    );

    ret(x) = Halide::Tuple(val, Halide::select(is_valid, 1, 0));

    return ret;
}

// -----------------------------------------------------------------------------
// High-level Set Operations (with automatic sorting)
// -----------------------------------------------------------------------------

/// @brief Compute intersection of two arrays (with automatic sorting)
/// @param a First array
/// @param b Second array
/// @param size_a Size of first array (must be power of 2)
/// @param size_b Size of second array (must be power of 2)
/// @param name Function name
/// @return Tuple: (intersection values, valid mask)
inline
Halide::Func intersect1d(Halide::Func a, Halide::Func b,
                          int size_a, int size_b,
                          std::string const& name = "intersect1d")
{
    auto sorted_a = bitonic_sort(a, size_a, name + "_sorted_a");
    auto sorted_b = bitonic_sort(b, size_b, name + "_sorted_b");
    sorted_a.compute_root();
    sorted_b.compute_root();

    return intersect1d_sorted(sorted_a, sorted_b, size_a, size_b, name);
}

/// @brief Compute set difference (with automatic sorting)
/// @param a First array
/// @param b Second array
/// @param size_a Size of first array (must be power of 2)
/// @param size_b Size of second array (must be power of 2)
/// @param name Function name
/// @return Tuple: (difference values, valid mask)
inline
Halide::Func setdiff1d(Halide::Func a, Halide::Func b,
                        int size_a, int size_b,
                        std::string const& name = "setdiff1d")
{
    auto sorted_a = bitonic_sort(a, size_a, name + "_sorted_a");
    auto sorted_b = bitonic_sort(b, size_b, name + "_sorted_b");
    sorted_a.compute_root();
    sorted_b.compute_root();

    return setdiff1d_sorted(sorted_a, sorted_b, size_a, size_b, name);
}

/// @brief Compute union of two arrays (with automatic sorting)
/// @param a First array
/// @param b Second array
/// @param size_a Size of first array (must be power of 2)
/// @param size_b Size of second array (must be power of 2)
/// @param name Function name
/// @return Tuple: (union values, valid mask) with size = size_a + size_b
inline
Halide::Func union1d(Halide::Func a, Halide::Func b,
                      int size_a, int size_b,
                      std::string const& name = "union1d")
{
    auto sorted_a = bitonic_sort(a, size_a, name + "_sorted_a");
    auto sorted_b = bitonic_sort(b, size_b, name + "_sorted_b");
    sorted_a.compute_root();
    sorted_b.compute_root();

    return union1d_sorted(sorted_a, sorted_b, size_a, size_b, name);
}

// -----------------------------------------------------------------------------
// Symmetric Difference and Membership
// -----------------------------------------------------------------------------

/// @brief Symmetric difference of two sorted arrays (elements in exactly one)
/// @param sorted_a First sorted 1D Func
/// @param sorted_b Second sorted 1D Func
/// @param size_a Size of first array
/// @param size_b Size of second array
/// @param name Function name
/// @return Tuple Func at positions [0..size_a+size_b-1]: (value, valid_mask)
///
/// Elements valid if in a but not b, OR in b but not a.
inline
Halide::Func setxor1d_sorted(Halide::Func sorted_a, Halide::Func sorted_b,
    int size_a, int size_b,
    std::string const& name = "setxor1d")
{
    // Elements in a not in b
    auto in_b_from_a = in1d(sorted_a, sorted_b, size_a, size_b, name + "_inb");
    auto marked_a    = mark_unique(sorted_a, size_a, name + "_marka");
    marked_a.compute_root();

    // Elements in b not in a
    auto in_a_from_b = in1d(sorted_b, sorted_a, size_b, size_a, name + "_ina");
    auto marked_b    = mark_unique(sorted_b, size_b, name + "_markb");
    marked_b.compute_root();

    Halide::Func ret(name);
    Halide::Var x("x");

    Halide::Expr in_a_section = x < size_a;

    Halide::Expr val = Halide::select(in_a_section, sorted_a(x), sorted_b(x - size_a));

    Halide::Expr is_valid = Halide::select(
        in_a_section,
        (marked_a(x)[1] == 1) && (in_b_from_a(x) == 0),
        (marked_b(x - size_a)[1] == 1) && (in_a_from_b(x - size_a) == 0)
    );

    ret(x) = Halide::Tuple(val, Halide::select(is_valid, 1, 0));
    return ret;
}

/// @brief Test element-wise membership: is elements(i) in test_against?
/// @param elements Values to test (can be 2D or any shape, tested element-wise)
/// @param test_against Values to test against (1D sorted array)
/// @param elem_shape Shape of elements array
/// @param against_size Size of test_against array
/// @param name Function name
/// @return Func of same shape as elements, 1 if element found in test_against, 0 otherwise
inline
Halide::Func isin(Halide::Func elements, Halide::Func test_against,
    const shape_t& elem_shape, int against_size,
    std::string const& name = "isin")
{
    // Flatten elements to 1D conceptually, do membership test, reshape back
    // Since Halide Funcs can be accessed at arbitrary coords, we just apply
    // the membership test at each point of elem_shape

    Halide::Func ret(name);
    std::vector<Halide::Var> vars;
    for (int d = 0; d < elem_shape.rank; ++d) vars.push_back(Halide::Var());

    Halide::Expr val = elements(vars);
    Halide::RDom r(0, against_size, "r_isin");

    // Check if val exists in test_against
    ret(vars) = Halide::maximum(Halide::select(test_against(r) == val, 1, 0));

    return ret;
}

NS_NUM_HALIDE_END
