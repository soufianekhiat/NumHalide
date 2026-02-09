/// @file manipulation_func.h
/// @brief Manipulation functions for Halide::Func objects
///
/// Provides: flatten, reshape, concat, vstack, hstack, dstack

#pragma once

#include "common.h"
#include "numhalide.h"
#include "shape.h"

NS_NUM_HALIDE_BEGIN

/// @brief Flatten a multidimensional Func into 1D
/// @param in Input Func
/// @param sizes Dimension sizes
/// @param name Function name
/// @return Flattened 1D Func
///
/// Usage:
///   Var x, y;
///   Func in; in(x, y) = x + y;
///   Func flat = flatten(in, {5, 4});
inline
Halide::Func flatten(Halide::Func in, std::vector<Halide::Expr> const& sizes, std::string const& name = "flatten")
{
    Halide::Func ret(name);
    Halide::Var x;
    std::vector<Halide::Expr> aArgs;
    index_to_args(aArgs, x, sizes);
    ret(x) = in(aArgs);
    return ret;
}

/// @brief Reshape a Func to new dimensions
/// @param in Input Func
/// @param sizes Original dimension sizes
/// @param new_sizes New dimension sizes
/// @param name Function name
/// @return Reshaped Func
///
/// Usage:
///   Func in; // 5x4 array
///   Func reshaped = reshape(in, {5,4}, {4,5});
inline
Halide::Func reshape(Halide::Func in, std::vector<Halide::Expr> const& sizes, std::vector<Halide::Expr> const& new_sizes, std::string const& name = "reshape")
{
    Halide::Func ret(name);
    std::vector<Halide::Var> vars;
    vars.resize(new_sizes.size());

    std::vector<Halide::Expr> varsVal;
    varsVal.reserve(sizes.size() + new_sizes.size());
    Halide::Expr prod0 = 1;
    for (Halide::Expr e : sizes) { prod0 *= e; varsVal.push_back(e); }
    Halide::Expr prod1 = 1;
    for (Halide::Expr e : new_sizes) { prod1 *= e; varsVal.push_back(e); }
    Halide::require(prod0 == prod1, varsVal);

    std::vector<Halide::Expr> exprVars;
    exprVars.resize(vars.size());
    std::transform(vars.begin(), vars.end(), exprVars.begin(), [](Halide::Var v){ return (Halide::Expr)v; });
    Halide::Expr idx = args_to_index(exprVars, new_sizes);

    Halide::Func flat = flatten(in, sizes);
    ret(vars) = flat(idx);
    return ret;
}

/// @brief Concatenate two Funcs along an axis
/// @param a First Func
/// @param shape_a Shape of first Func
/// @param b Second Func
/// @param shape_b Shape of second Func
/// @param axis Axis to concatenate along
/// @param name Function name
/// @return Concatenated Func
inline
Halide::Func concat(Halide::Func a, const shape_t& shape_a, Halide::Func b, const shape_t& shape_b, int axis, std::string const& name = "concat")
{
    int norm_axis = normalized_axis(axis, shape_a.rank);
    nh_require(nullptr, check_same_except(shape_a, shape_b, norm_axis),
        "Shapes %s and %s mismatch for concat at axis %d",
        shape_to_string(shape_a).c_str(), shape_to_string(shape_b).c_str(), axis);

    Halide::Func ret(name);
    std::vector<Halide::Var> vars;
    for (int i = 0; i < shape_a.rank; ++i) {
        vars.push_back(Halide::Var());
    }

    // norm_axis is the index in shape (0..rank-1).
    // The corresponding Halide Var index is rank - 1 - norm_axis.
    int var_idx = shape_a.rank - 1 - norm_axis;
    Halide::Expr split_point = shape_a.extents[norm_axis];
    Halide::Var split_var = vars[var_idx];

    std::vector<Halide::Expr> args_b;
    // b expects args in order [dim_rank-1, ..., dim_0] (inner to outer)
    // We iterate k from rank-1 down to 0.
    for (int k = shape_a.rank - 1; k >= 0; --k) {
        int current_var_idx = shape_a.rank - 1 - k;
        if (k == norm_axis) {
            args_b.push_back(vars[current_var_idx] - split_point);
        } else {
            args_b.push_back(vars[current_var_idx]);
        }
    }

    // Ensure boolean condition for select
    Halide::Expr cond = split_var < split_point;
    // Halide::select requires Expr arguments, a(vars) and b(args_b) return FuncRef/Expr
    ret(vars) = Halide::select(cond, a(vars), b(args_b));
    return ret;
}

/// @brief Stack arrays vertically (along axis 0)
inline
Halide::Func vstack(Halide::Func a, const shape_t& shape_a, Halide::Func b, const shape_t& shape_b, std::string const& name = "vstack")
{
    if (shape_a.rank == 1 && shape_b.rank == 1) {
        nh_require(nullptr, shape_a.extents[0] == shape_b.extents[0],
            "Shapes %s and %s mismatch for vstack (dim 1 must match)",
            shape_to_string(shape_a).c_str(), shape_to_string(shape_b).c_str());
        Halide::Func ret(name);
        Halide::Var x("x"), y("y");
        // Row 0 from a, Row 1 from b
        ret(x, y) = Halide::select(y < 1, a(x), b(x));
        return ret;
    }
    return concat(a, shape_a, b, shape_b, 0, name);
}

/// @brief Stack arrays horizontally (along axis 1)
inline
Halide::Func hstack(Halide::Func a, const shape_t& shape_a, Halide::Func b, const shape_t& shape_b, std::string const& name = "hstack")
{
    // Special case for 2D arrays: stack horizontally (columns)
    if (shape_a.rank == 2 && shape_b.rank == 2 && shape_a.extents[0] == shape_b.extents[0]) {
        // Ensure same number of rows
        nh_require(nullptr, shape_a.extents[0] == shape_b.extents[0],
            "Shapes %s and %s mismatch for hstack (rows must match)",
            shape_to_string(shape_a).c_str(), shape_to_string(shape_b).c_str());
        Halide::Func ret(name);
        Halide::Var y("y"), x("x");
        Halide::Expr split_point = shape_a.extents[1]; // columns of a
        // Row index y, column index x
        ret(x, y) = Halide::select(x < split_point, a(x, y), b(x - split_point, y));
        return ret;
    }
    return concat(a, shape_a, b, shape_b, 1, name);
}

/// @brief Stack arrays depth‑wise (along axis 2)
inline
Halide::Func dstack(Halide::Func a, const shape_t& shape_a, Halide::Func b, const shape_t& shape_b, std::string const& name = "dstack")
{
    return concat(a, shape_a, b, shape_b, 2, name);
}

NS_NUM_HALIDE_END
