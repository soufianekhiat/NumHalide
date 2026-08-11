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
// Power-of-2 Utilities
// -----------------------------------------------------------------------------

/// @brief Check if n is a power of 2
inline bool is_power_of_2(int n) { return n > 0 && (n & (n - 1)) == 0; }

/// @brief Return smallest power of 2 >= n
inline int next_power_of_2(int n) {
    if (n <= 1) return 1;
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

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
Halide::Func mark_unique(Halide::Func sorted_input, int /*size*/, std::string const& name = "mark_unique")
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
    nh_require((size & (size - 1)) == 0, "unique requires power of 2 size, got %d", size);

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
                  int /*test_size*/, int array_size,
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
    nh_require(is_power_of_2(size_a),
        "intersect1d: size_a must be a power of 2, got %d (next valid size: %d)",
        size_a, next_power_of_2(size_a));
    nh_require(is_power_of_2(size_b),
        "intersect1d: size_b must be a power of 2, got %d (next valid size: %d)",
        size_b, next_power_of_2(size_b));
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
    nh_require(is_power_of_2(size_a),
        "setdiff1d: size_a must be a power of 2, got %d (next valid size: %d)",
        size_a, next_power_of_2(size_a));
    nh_require(is_power_of_2(size_b),
        "setdiff1d: size_b must be a power of 2, got %d (next valid size: %d)",
        size_b, next_power_of_2(size_b));
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
    nh_require(is_power_of_2(size_a),
        "union1d: size_a must be a power of 2, got %d (next valid size: %d)",
        size_a, next_power_of_2(size_a));
    nh_require(is_power_of_2(size_b),
        "union1d: size_b must be a power of 2, got %d (next valid size: %d)",
        size_b, next_power_of_2(size_b));
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

// -----------------------------------------------------------------------------
// Fixed-Size Positional / Zero-Fill Variants (RUNTIME sizes)
// -----------------------------------------------------------------------------
//
// The functions below are the FIXED-SIZE positional encodings of the set
// operations: the output has the same length as the (first) input, and set
// membership is expressed in place — by keeping a value at its position or
// writing 0 (or an Int32 0/1 mark). They exist for frameworks whose buffers
// cannot change size at runtime; sizes are Halide::Expr, and the element
// type is whatever the input Func carries (integer types included — no
// float promotion happens here). The mask-Tuple forms above remain the
// variable-size-style API (value, valid-mask); these do not replace them.

/// @brief Count unique elements in a sorted array, RUNTIME size (0-D output)
/// @param sorted_input Sorted 1D Func (any element type)
/// @param n Number of elements as a runtime expression
/// @param name Function name
/// @return 0-D Int32 Func: 1 + number of adjacent transitions in the input
///
/// count = 1 + #{ r in [1, n) : sorted_input(r) != sorted_input(r-1) }.
/// For n >= 1 this is the unique-element count of the sorted array. For
/// n == 0 the reduction domain is empty and the result is still 1 — this
/// fixed-size encoding has no way to signal an empty input, so callers must
/// treat n == 0 as out of contract.
inline
Halide::Func count_unique(Halide::Func sorted_input, Halide::Expr n,
                          std::string const& name = "count_unique_rt")
{
    Halide::Func ret(name);
    Halide::RDom r(1, Halide::max(n - 1, 0), "r_" + name);

    Halide::Expr is_diff =
        Halide::cast<int32_t>(sorted_input(r) != sorted_input(r - 1));

    ret() = Halide::cast<int32_t>(1) + Halide::sum(is_diff, "s_" + name);
    return ret;
}

/// @brief Mark first occurrences in a sorted array, RUNTIME size
/// @param sorted_input Sorted 1D Func (any element type)
/// @param n Number of elements as a runtime expression
/// @param name Function name
/// @return 1-D Int32 Func: ret(0) = 1; ret(i) = sorted_input(i) != sorted_input(i-1)
///
/// Unlike the compile-time Tuple overload above, this returns the Int32
/// marks ONLY (the values stay in the input). Both neighbor reads are
/// clamped to [0, n-1] UNCONDITIONALLY — the clamp is not guarded by the
/// select condition, so no simplification can strip it and turn the
/// speculative i-1 read into an out-of-bounds access.
inline
Halide::Func mark_unique(Halide::Func sorted_input, Halide::Expr n,
                         std::string const& name = "mark_unique_rt")
{
    Halide::Func ret(name);
    Halide::Var x("x");

    Halide::Expr cur  = sorted_input(Halide::clamp(x,     0, n - 1));
    Halide::Expr prev = sorted_input(Halide::clamp(x - 1, 0, n - 1));

    // ret[0] = 1 (first element always unique)
    // ret[i] = 1 if sorted_input[i] != sorted_input[i-1], else 0
    ret(x) = Halide::select(x == 0,
                            Halide::cast<int32_t>(1),
                            Halide::select(cur != prev,
                                           Halide::cast<int32_t>(1),
                                           Halide::cast<int32_t>(0)));
    return ret;
}

/// @brief Unique elements of a sorted array, zero-fill positional form
/// @param sorted_input Sorted 1D Func (any element type)
/// @param n Number of elements as a runtime expression
/// @param name Function name
/// @return 1-D Func, same element type: sorted_input(i) at first occurrences,
///         0 at duplicate positions
///
/// FIXED-SIZE variant of unique(): instead of compacting (which needs a
/// variable-size output), duplicates are zeroed in place. The mask-Tuple /
/// pad-with-last unique() above remains the variable-size-style API.
inline
Halide::Func unique_zerofill(Halide::Func sorted_input, Halide::Expr n,
                             std::string const& name = "unique_zerofill")
{
    Halide::Func ret(name);
    Halide::Var x("x");

    Halide::Expr cur  = sorted_input(Halide::clamp(x,     0, n - 1));
    Halide::Expr prev = sorted_input(Halide::clamp(x - 1, 0, n - 1));

    // Keep value if first occurrence, zero it out if duplicate
    ret(x) = Halide::select(x == 0,
                            cur,
                            Halide::select(cur != prev,
                                           cur,
                                           Halide::cast(cur.type(), 0)));
    return ret;
}

/// @brief Element-wise membership test, RUNTIME set size
/// @param values 1D Func of query values (any element type)
/// @param test_set 1D Func of set elements (same element type)
/// @param n_set Number of elements in test_set as a runtime expression
/// @param name Function name
/// @return 1-D Int32 Func: 1 if values(i) occurs anywhere in test_set, else 0
///
/// Linear count form: ret(i) = (#{ r : values(i) == test_set(r) } > 0).
/// The scan never exploits ordering, so test_set may be sorted or unsorted —
/// both give identical results.
inline
Halide::Func in1d(Halide::Func values, Halide::Func test_set,
                  Halide::Expr n_set,
                  std::string const& name = "in1d_rt")
{
    Halide::Func ret(name);
    Halide::Var i("i");
    Halide::RDom r(0, Halide::max(n_set, 0), "r_" + name);

    ret(i) = Halide::cast<int32_t>(
        Halide::sum(Halide::cast<int32_t>(values(i) == test_set(r)),
                    "s_" + name) > 0);
    return ret;
}

/// @brief Intersection, zero-fill positional form (RUNTIME size of b)
/// @param a First 1D Func (any element type)
/// @param b Second 1D Func (same element type)
/// @param n_b Number of elements in b as a runtime expression
/// @param name Function name
/// @return 1-D Func, same element type and length as a:
///         a(i) if a(i) occurs in b, else 0
///
/// FIXED-SIZE variant of intersect1d: membership is positional (keep or
/// zero), the output length equals |a|, and the linear scan works for
/// sorted and unsorted inputs alike.
inline
Halide::Func intersect1d_zerofill(Halide::Func a, Halide::Func b,
                                  Halide::Expr n_b,
                                  std::string const& name = "intersect1d_zerofill")
{
    Halide::Func ret(name);
    Halide::Var i("i");
    Halide::RDom r(0, Halide::max(n_b, 0), "r_" + name);

    Halide::Expr found = Halide::cast<int32_t>(
        Halide::sum(Halide::cast<int32_t>(a(i) == b(r)), "s_" + name) > 0);

    Halide::Expr a_val = a(i);
    ret(i) = Halide::select(found > 0, a_val, Halide::cast(a_val.type(), 0));
    return ret;
}

/// @brief Set difference (a \ b), zero-fill positional form (RUNTIME size of b)
/// @param a First 1D Func (any element type)
/// @param b Second 1D Func (same element type)
/// @param n_b Number of elements in b as a runtime expression
/// @param name Function name
/// @return 1-D Func, same element type and length as a:
///         0 if a(i) occurs in b, else a(i)
///
/// FIXED-SIZE variant of setdiff1d: the complement of intersect1d_zerofill.
/// Works for sorted and unsorted inputs alike.
inline
Halide::Func setdiff1d_zerofill(Halide::Func a, Halide::Func b,
                                Halide::Expr n_b,
                                std::string const& name = "setdiff1d_zerofill")
{
    Halide::Func ret(name);
    Halide::Var i("i");
    Halide::RDom r(0, Halide::max(n_b, 0), "r_" + name);

    Halide::Expr in_b = Halide::cast<int32_t>(
        Halide::sum(Halide::cast<int32_t>(a(i) == b(r)), "s_" + name) > 0);

    Halide::Expr a_val = a(i);
    ret(i) = Halide::select(in_b > 0, Halide::cast(a_val.type(), 0), a_val);
    return ret;
}

/// @brief Positional union: element-wise maximum of two same-length arrays
/// @param a First 1D Func (any element type)
/// @param b Second 1D Func (same element type and length)
/// @param name Function name
/// @return 1-D Func, same element type: max(a(i), b(i))
///
/// FIXED-SIZE stand-in for a set union: a true union needs a variable-size
/// output, so this positional encoding takes the element-wise maximum of
/// two equal-length arrays instead. No size argument is needed — the
/// operation is pointwise.
inline
Halide::Func union1d_positional(Halide::Func a, Halide::Func b,
                                std::string const& name = "union1d_positional")
{
    Halide::Func ret(name);
    Halide::Var i("i");
    ret(i) = Halide::max(a(i), b(i));
    return ret;
}

NS_NUM_HALIDE_END
