/// @file pad_func.h
/// @brief Array padding with various boundary modes

#pragma once
#include "numhalide.h"
#include "shape.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// 1D padding
// -----------------------------------------------------------------------------

/// @brief Pad a 1D array with a constant value
/// @param f      Input Func of size n
/// @param n      Size of input
/// @param before Elements prepended (filled with value)
/// @param after  Elements appended  (filled with value)
/// @return Func of size n + before + after
inline Halide::Func pad(Halide::Func f, int n, int before, int after,
    float value = 0.0f,
    std::string const& name = "pad")
{
    (void)after;
    Halide::Var i("i");
    Halide::Expr orig = i - before;
    Halide::Func ret(name);
    // Guard as multiplied 0/1 factors + unconditional clamp — a select whose
    // condition proves the clamp redundant lets the simplifier strip it and
    // CMOV reads OOB (see polymul in polynomial.h).
    Halide::Type type = f.types()[0];
    Halide::Expr valid = orig >= 0 && orig < n;
    ret(i) = f(Halide::clamp(orig, 0, n - 1)) * Halide::cast(type, valid)
        + Halide::cast(type, value) * Halide::cast(type, !valid);
    return ret;
}

/// @brief Pad a 1D array by replicating the edge values
/// @details Elements beyond the boundary take the value of the nearest edge.
///          pad_edge([1,2,3], 2, 2) → [1,1, 1,2,3, 3,3]
inline Halide::Func pad_edge(Halide::Func f, int n, int before, int after,
    std::string const& name = "pad_edge")
{
    (void)after;
    Halide::Var i("i");
    Halide::Expr orig = i - before;
    Halide::Func ret(name);
    ret(i) = f(Halide::clamp(orig, 0, n - 1));
    return ret;
}

/// @brief Pad a 1D array by reflecting values at the boundary (boundary not repeated)
/// @details np.pad mode='reflect'.
///          pad_reflect([1,2,3], 2, 2) → [3,2, 1,2,3, 2,1]
inline Halide::Func pad_reflect(Halide::Func f, int n, int before, int after,
    std::string const& name = "pad_reflect")
{
    (void)after;
    Halide::Var i("i");
    Halide::Expr orig = i - before;
    // Mirror at 0: orig<0 → -orig
    // Mirror at n-1: orig>=n → 2*(n-1) - orig
    Halide::Expr idx = Halide::select(
        orig < 0,   -orig,
        Halide::select(orig >= n, 2*(n - 1) - orig, orig));
    Halide::Func ret(name);
    ret(i) = f(Halide::clamp(idx, 0, n - 1));
    return ret;
}

/// @brief Pad a 1D array by wrapping values circularly
/// @details np.pad mode='wrap'.
///          pad_wrap([1,2,3], 2, 2) → [2,3, 1,2,3, 1,2]
inline Halide::Func pad_wrap(Halide::Func f, int n, int before, int after,
    std::string const& name = "pad_wrap")
{
    (void)after;
    Halide::Var i("i");
    Halide::Expr orig = i - before;
    // Correct modulo for negative values: ((orig % n) + n) % n
    Halide::Expr idx = ((orig % n) + n) % n;
    Halide::Func ret(name);
    ret(i) = f(idx);
    return ret;
}

// -----------------------------------------------------------------------------
// 2D padding
// -----------------------------------------------------------------------------

/// @brief Pad a 2D array with a constant value
/// @param shape  {rows, cols} of input
/// @param top    Rows prepended at top
/// @param bottom Rows appended at bottom
/// @param left   Cols prepended at left
/// @param right  Cols appended at right
/// @return Func of shape (rows+top+bottom, cols+left+right)
inline Halide::Func pad_2d(Halide::Func f, const shape_t& shape,
    int top, int bottom, int left, int right,
    float value = 0.0f,
    std::string const& name = "pad2d")
{
    (void)bottom; (void)right;
    int rows = shape[0], cols = shape[1];
    Halide::Var x("x"), y("y");
    Halide::Expr ox = x - left;
    Halide::Expr oy = y - top;
    Halide::Func ret(name);
    // Guard as multiplied 0/1 factors + unconditional clamps — a select whose
    // condition proves a clamp redundant lets the simplifier strip it and
    // CMOV reads OOB (see polymul in polynomial.h).
    Halide::Type type = f.types()[0];
    Halide::Expr valid = ox >= 0 && ox < cols && oy >= 0 && oy < rows;
    ret(x, y) = f(Halide::clamp(ox, 0, cols - 1), Halide::clamp(oy, 0, rows - 1))
            * Halide::cast(type, valid)
        + Halide::cast(type, value) * Halide::cast(type, !valid);
    return ret;
}

/// @brief Pad a 2D array by replicating the edge values
inline Halide::Func pad_2d_edge(Halide::Func f, const shape_t& shape,
    int top, int bottom, int left, int right,
    std::string const& name = "pad2d_edge")
{
    (void)bottom; (void)right;
    int rows = shape[0], cols = shape[1];
    Halide::Var x("x"), y("y");
    Halide::Func ret(name);
    ret(x, y) = f(Halide::clamp(x - left, 0, cols - 1),
                  Halide::clamp(y - top,  0, rows - 1));
    return ret;
}

/// @brief Pad a 2D array by reflecting at the boundary (np.pad mode='reflect')
inline Halide::Func pad_2d_reflect(Halide::Func f, const shape_t& shape,
    int top, int bottom, int left, int right,
    std::string const& name = "pad2d_reflect")
{
    (void)bottom; (void)right;
    int rows = shape[0], cols = shape[1];
    Halide::Var x("x"), y("y");
    Halide::Expr ox = x - left;
    Halide::Expr oy = y - top;
    Halide::Expr ix = Halide::select(ox < 0, -ox, Halide::select(ox >= cols, 2*(cols-1)-ox, ox));
    Halide::Expr iy = Halide::select(oy < 0, -oy, Halide::select(oy >= rows, 2*(rows-1)-oy, oy));
    Halide::Func ret(name);
    ret(x, y) = f(Halide::clamp(ix, 0, cols-1), Halide::clamp(iy, 0, rows-1));
    return ret;
}

/// @brief Pad a 2D array by wrapping circularly (np.pad mode='wrap')
inline Halide::Func pad_2d_wrap(Halide::Func f, const shape_t& shape,
    int top, int bottom, int left, int right,
    std::string const& name = "pad2d_wrap")
{
    (void)bottom; (void)right;
    int rows = shape[0], cols = shape[1];
    Halide::Var x("x"), y("y");
    Halide::Expr ox = x - left;
    Halide::Expr oy = y - top;
    Halide::Expr ix = ((ox % cols) + cols) % cols;
    Halide::Expr iy = ((oy % rows) + rows) % rows;
    Halide::Func ret(name);
    ret(x, y) = f(ix, iy);
    return ret;
}

NS_NUM_HALIDE_END
