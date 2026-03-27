/// @file flip_roll.h
/// @brief Flip and roll (circular shift) operations

#pragma once
#include "numhalide.h"
#include "shape.h"

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// flip — reverse elements along an axis
// -----------------------------------------------------------------------------

/// @brief Reverse a 1D array of n elements
inline Halide::Func flip(Halide::Func f, int n,
    std::string const& name = "flip")
{
    Halide::Var i("i");
    Halide::Func ret(name);
    ret(i) = f(n - 1 - i);
    return ret;
}

/// @brief Flip a 2D array left-right (reverse columns, axis=1)
inline Halide::Func fliplr(Halide::Func f, int n_cols,
    std::string const& name = "fliplr")
{
    Halide::Var x("x"), y("y");
    Halide::Func ret(name);
    ret(x, y) = f(n_cols - 1 - x, y);
    return ret;
}

/// @brief Flip a 2D array up-down (reverse rows, axis=0)
inline Halide::Func flipud(Halide::Func f, int n_rows,
    std::string const& name = "flipud")
{
    Halide::Var x("x"), y("y");
    Halide::Func ret(name);
    ret(x, y) = f(x, n_rows - 1 - y);
    return ret;
}

// -----------------------------------------------------------------------------
// roll — circular shift
// -----------------------------------------------------------------------------

/// @brief Roll a 1D array of size n by `shift` positions
/// @details Positive shift moves elements toward higher indices (right).
///          roll([1,2,3,4,5], 2) → [4,5,1,2,3]
inline Halide::Func roll(Halide::Func f, int n, int shift,
    std::string const& name = "roll")
{
    // Normalize shift to [0, n)
    int s = ((shift % n) + n) % n;
    Halide::Var i("i");
    Halide::Func ret(name);
    // result[i] = f[(i - s + n) % n]
    // (i - s + n) is always in [1, 2n-1] for i in [0,n-1] and s in [0,n-1]
    // so a single modulo is sufficient
    ret(i) = f((i - s + n) % n);
    return ret;
}

NS_NUM_HALIDE_END
