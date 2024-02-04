#pragma once

#include "NumHalide/Common.h"
#include "NumHalide/NumHalide.h"

NS_NUM_HALIDE_FUNC_BEGIN

//////////////////////////////////////////////////////////////////////////
// Usage:
//	Var x, y;
//	Buffer<f32> values( Float( 32 ), { 5, 4 } );
//	Func in;
//	in( x, y ) = values( x, y );
//	Func flat = flatten( in, { 5, 4 }, "flatten" );
//	RDom r( 0, 5*4 );
//	Expr total = sum( flat( r ) );
//////////////////////////////////////////////////////////////////////////
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

//////////////////////////////////////////////////////////////////////////
// Usage:
//	Var x, y;
//	Buffer<f32> values( Float( 32 ), { 5, 4 } );
//	Func in;
//	in( x, y ) = values( x, y );
//	Func flat = flatten( in, { 5, 4 }, "flatten" );
//	RDom r( 0, 5*4 );
//	Expr total = sum( flat( r ) );
//////////////////////////////////////////////////////////////////////////
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
	// TODO
}

inline
Func hstack(std::vector<Func> const& tup)
{
	// TODO
}

inline
Func dstack(std::vector<Func> const& tup)
{
	// TODO
}

NS_NUM_HALIDE_FUNC_END
