/// @file manipulation_func.h
/// @brief Manipulation functions for Halide::Func objects
///
/// Provides: flatten, reshape, vstack, hstack, dstack

#pragma once

#include "common.h"
#include "numhalide.h"

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
	for (Halide::Expr e : sizes)
	{
		prod0 *= e;
		varsVal.push_back(e);
	}
	Halide::Expr prod1 = 1;
	for (Halide::Expr e : new_sizes)
	{
		prod1 *= e;
		varsVal.push_back(e);
	}

	Halide::require(prod0 == prod1, varsVal);

	std::vector<Halide::Expr> exprVars;
	exprVars.resize(vars.size());
	std::transform(vars.begin(), vars.end(), exprVars.begin(),
		[](Halide::Var v)
		{
			return (Halide::Expr)v;
		});
	Halide::Expr idx = args_to_index(exprVars, new_sizes);

	Halide::Func flat = flatten(in, sizes);

	ret(vars) = flat(idx);

	return ret;
}

/// @brief Stack arrays vertically (along axis 0)
/// @param tup Vector of Funcs to stack
/// @return Vertically stacked Func
inline
Halide::Func vstack(std::vector<Halide::Func> const& tup)
{
	// TODO: Implement vstack
	NH_ASSERT(false && "vstack not yet implemented");
	return Halide::Func();
}

/// @brief Stack arrays horizontally (along axis 1)
/// @param tup Vector of Funcs to stack
/// @return Horizontally stacked Func
inline
Halide::Func hstack(std::vector<Halide::Func> const& tup)
{
	// TODO: Implement hstack
	NH_ASSERT(false && "hstack not yet implemented");
	return Halide::Func();
}

/// @brief Stack arrays depth-wise (along axis 2)
/// @param tup Vector of Funcs to stack
/// @return Depth-stacked Func
inline
Halide::Func dstack(std::vector<Halide::Func> const& tup)
{
	// TODO: Implement dstack
	NH_ASSERT(false && "dstack not yet implemented");
	return Halide::Func();
}

NS_NUM_HALIDE_END
