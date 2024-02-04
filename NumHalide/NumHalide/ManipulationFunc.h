#pragma once

#include "NumHalide/Common.h"
#include "NumHalide/NumHalide.h"

NS_NUM_HALIDE_FUNC_BEGIN

inline
Func flatten(Func in, std::vector<Expr> const& sizes, std::string const& name = "flatten")
{
	Func ret(name);
	Var x;

	std::vector< Expr > aArgs;
	index_to_args(aArgs, x, sizes);

	ret(x) = in(aArgs);

	return ret;
}

inline
Func reshape(Func in, std::vector<Expr> const& sizes, std::vector<Expr> const& new_sizes, std::string const& name = "reshape")
{
	Func ret(name);
	std::vector<Var> vars;
	vars.resize(new_sizes.size());

	std::vector<Expr> varsVal;
	varsVal.reserve(sizes.size() + new_sizes.size());
	Expr prod0 = 1;
	for (Expr e : sizes)
	{
		prod0 *= e;
		varsVal.push_back(e);
	}
	Expr prod1 = 1;
	for (Expr e : new_sizes)
	{
		prod1 *= e;
		varsVal.push_back(e);
	}

	require(prod0 == prod1, varsVal);

	std::vector<Expr> exprVars;
	exprVars.resize(vars.size());
	std::transform( vars.begin(), vars.end(), exprVars.begin(),
		[]( Var v )
		{
			return (Expr)v;
		});
	Expr idx = args_to_index(exprVars, new_sizes);

	Func flat = flatten(in, sizes);

	ret(vars) = flat(idx);

	return ret;
}

inline
Func vstack(std::vector<Func> const& tup)
{
}

inline
Func hstack(std::vector<Func> const& tup)
{
}

inline
Func dstack(std::vector<Func> const& tup)
{
}

NS_NUM_HALIDE_FUNC_END
