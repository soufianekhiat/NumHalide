/// @file broadcast.h
/// @brief Broadcasting and elementwise operations
///
/// Provides: broadcast_to, unary ops, binary ops with broadcasting

#pragma once

#include "common.h"
#include "numhalide.h"
#include "shape.h"
#include <algorithm>

NS_NUM_HALIDE_BEGIN

// -----------------------------------------------------------------------------
// Broadcasting
// -----------------------------------------------------------------------------

/// @brief Broadcast a Func to a target shape
/// @param f Input Func
/// @param src_shape Input shape
/// @param dst_shape Target shape
/// @param name Function name
/// @return Broadcasted Func
inline Halide::Func broadcast_to(Halide::Func f, const shape_t& src_shape, const shape_t& dst_shape, std::string const& name = "broadcast_to") {
    int offset = dst_shape.rank - src_shape.rank;
    nh_require(nullptr, offset >= 0, "Target rank %d must be >= source rank %d", dst_shape.rank, src_shape.rank);

    Halide::Func ret(name);
    std::vector<Halide::Var> vars;
    for (int i = 0; i < dst_shape.rank; ++i) vars.push_back(Halide::Var());

    std::vector<Halide::Expr> args;
    args.resize(src_shape.rank);
    // Map source dimensions to destination variables
    // Shape convention: extents[0] is outermost, extents[rank-1] is innermost
    // Halide convention: args[0] is innermost, args[rank-1] is outermost
    for (int i = 0; i < src_shape.rank; ++i) {
        int src_dim = src_shape.extents[i];
        int dst_idx = i + offset; // index in dst_shape
        int dst_dim = dst_shape.extents[dst_idx];
        // Halide Var ordering: vars[0] is innermost, vars[rank-1] outermost
        int var_idx = dst_shape.rank - 1 - dst_idx;
        // Place in args at position matching Halide convention (inner first)
        int arg_idx = src_shape.rank - 1 - i;
        if (src_dim == dst_dim) {
            args[arg_idx] = vars[var_idx];
        } else if (src_dim == 1) {
            args[arg_idx] = 0;
        } else {
            nh_require(nullptr, false, "Cannot broadcast dim %d of size %d to %d", i, src_dim, dst_dim);
        }
    }

    ret(vars) = f(args);
    return ret;
}

// -----------------------------------------------------------------------------
// Unary Operations
// -----------------------------------------------------------------------------

#define NH_UNARY_OP(op_name, expr) \
inline Halide::Func op_name(Halide::Func a, const shape_t& shape, std::string const& name = #op_name) { \
    Halide::Func ret(name); \
    std::vector<Halide::Var> vars; \
    for (int i = 0; i < shape.rank; ++i) vars.push_back(Halide::Var()); \
    ret(vars) = expr; \
    return ret; \
}

NH_UNARY_OP(negative, -a(vars))
NH_UNARY_OP(abs, Halide::abs(a(vars)))
NH_UNARY_OP(exp, Halide::exp(a(vars)))
NH_UNARY_OP(log, Halide::log(a(vars)))
NH_UNARY_OP(sqrt, Halide::sqrt(a(vars)))

#undef NH_UNARY_OP

// -----------------------------------------------------------------------------
// Binary Operations
// -----------------------------------------------------------------------------

#define NH_BINARY_OP(op_name, expr) \
inline Halide::Func op_name(Halide::Func a, const shape_t& sa, Halide::Func b, const shape_t& sb, std::string const& name = #op_name) { \
    shape_t out_shape = infer_broadcast(sa, sb); \
    Halide::Func a_b = broadcast_to(a, sa, out_shape, name + "_a_bcast"); \
    Halide::Func b_b = broadcast_to(b, sb, out_shape, name + "_b_bcast"); \
    Halide::Func ret(name); \
    std::vector<Halide::Var> vars; \
    for (int i = 0; i < out_shape.rank; ++i) vars.push_back(Halide::Var()); \
    ret(vars) = expr; \
    return ret; \
}

// Arithmetic
NH_BINARY_OP(add, a_b(vars) + b_b(vars))
NH_BINARY_OP(sub, a_b(vars) - b_b(vars))
NH_BINARY_OP(mul, a_b(vars) * b_b(vars))
NH_BINARY_OP(div, a_b(vars) / b_b(vars))
NH_BINARY_OP(pow, Halide::pow(a_b(vars), b_b(vars)))
NH_BINARY_OP(minimum, Halide::min(a_b(vars), b_b(vars)))
NH_BINARY_OP(maximum, Halide::max(a_b(vars), b_b(vars)))

// Comparison
NH_BINARY_OP(equal, a_b(vars) == b_b(vars))
NH_BINARY_OP(not_equal, a_b(vars) != b_b(vars))
NH_BINARY_OP(less, a_b(vars) < b_b(vars))
NH_BINARY_OP(less_equal, a_b(vars) <= b_b(vars))
NH_BINARY_OP(greater, a_b(vars) > b_b(vars))
NH_BINARY_OP(greater_equal, a_b(vars) >= b_b(vars))

// Logical
NH_BINARY_OP(logical_and, a_b(vars) && b_b(vars))
NH_BINARY_OP(logical_or, a_b(vars) || b_b(vars))
NH_BINARY_OP(logical_xor, a_b(vars) != b_b(vars)) // XOR for boolean is !=

#undef NH_BINARY_OP

NS_NUM_HALIDE_END
