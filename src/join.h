/// @file join.h
/// @brief Concatenate and stack operations (inverse of split)

#pragma once
#include "numhalide.h"
#include "shape.h"
#include <vector>

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// concat_1d — join two 1D arrays end-to-end
// -----------------------------------------------------------------------------

/// @brief Concatenate two 1D Funcs into one of size n1+n2
inline Halide::Func concat_1d(Halide::Func f1, int n1,
    Halide::Func f2, int n2,
    std::string const& name = "concat")
{
    Halide::Var i("i");
    Halide::Func ret(name);
    if (n1 == 0) {
        ret(i) = f2(i);
    } else if (n2 == 0) {
        ret(i) = f1(i);
    } else {
        // Clamp both accesses so Halide's conservative bounds inference succeeds.
        // select still picks the correct branch; clamp only prevents out-of-bounds JIT errors.
        ret(i) = Halide::select(i < n1,
            f1(Halide::clamp(i, 0, n1 - 1)),
            f2(Halide::clamp(i - n1, 0, n2 - 1)));
    }
    return ret;
}

/// @brief Concatenate a list of 1D Funcs with given sizes into one array
inline Halide::Func concatenate(const std::vector<Halide::Func>& funcs,
    const std::vector<int>& sizes,
    std::string const& name = "cat")
{
    assert(!funcs.empty() && funcs.size() == sizes.size());

    // Build cumulative offsets
    std::vector<int> off(sizes.size(), 0);
    for (int k = 1; k < (int)sizes.size(); ++k)
        off[k] = off[k-1] + sizes[k-1];

    Halide::Var i("i");
    // Build nested selects right-to-left.
    // Clamp each access so bounds inference never sees out-of-range indices.
    int last = (int)funcs.size() - 1;
    Halide::Expr val = funcs[last](Halide::clamp(i - off[last], 0, sizes[last] - 1));
    for (int k = last - 1; k >= 0; --k)
        val = Halide::select(i < off[k] + sizes[k],
            funcs[k](Halide::clamp(i - off[k], 0, sizes[k] - 1)),
            val);

    Halide::Func ret(name);
    ret(i) = val;
    return ret;
}

// -----------------------------------------------------------------------------
// concat_2d — join two 2D Funcs along an axis
// -----------------------------------------------------------------------------

/// @brief Concatenate two 2D Funcs along axis 0 (rows) or axis 1 (columns)
/// @param f1   First Func: shape s1
/// @param s1   Shape of f1 — s1[0]=rows, s1[1]=cols
/// @param f2   Second Func: shape s2 (must match on the other axis)
/// @param s2   Shape of f2
/// @param axis 0 = stack rows (s1[1] must equal s2[1])
///             1 = stack cols (s1[0] must equal s2[0])
inline Halide::Func concat_2d(Halide::Func f1, const shape_t& s1,
    Halide::Func f2, const shape_t& s2,
    int axis,
    std::string const& name = "concat2d")
{
    (void)s2;
    Halide::Var x("x"), y("y");
    Halide::Func ret(name);
    if (axis == 0)
        ret(x, y) = Halide::select(y < s1[0],
            f1(x, Halide::clamp(y,          0, s1[0] - 1)),
            f2(x, Halide::clamp(y - s1[0],  0, s2[0] - 1)));
    else
        ret(x, y) = Halide::select(x < s1[1],
            f1(Halide::clamp(x,          0, s1[1] - 1), y),
            f2(Halide::clamp(x - s1[1],  0, s2[1] - 1), y));
    return ret;
}

// -----------------------------------------------------------------------------
// stack — stack N 1D arrays into a 2D array along a new axis
// -----------------------------------------------------------------------------

/// @brief Stack N 1D Funcs (each of length m) into a 2D Func
/// @param funcs  Vector of 1D Funcs, each of length m
/// @param m      Length of each input array
/// @param axis   0 → output shape (n_funcs × m): ret(col, row) where row picks array
///               1 → output shape (m × n_funcs): ret(col, row) where col picks array
///
/// Example (axis=0): stack([a,b,c], 4) → 3×4 matrix; row 0 = a, row 1 = b, row 2 = c
inline Halide::Func stack(const std::vector<Halide::Func>& funcs, int m,
    int axis = 0,
    std::string const& name = "stack")
{
    (void)m;
    Halide::Var x("x"), y("y");
    Halide::Func ret(name);

    if (axis == 0) {
        // y selects which array; x is the within-array index
        Halide::Expr val = funcs.back()(x);
        for (int j = (int)funcs.size() - 2; j >= 0; --j)
            val = Halide::select(y == j, funcs[j](x), val);
        ret(x, y) = val;
    } else {
        // x selects which array; y is the within-array index
        Halide::Expr val = funcs.back()(y);
        for (int j = (int)funcs.size() - 2; j >= 0; --j)
            val = Halide::select(x == j, funcs[j](y), val);
        ret(x, y) = val;
    }
    return ret;
}

NS_NUM_HALIDE_END
