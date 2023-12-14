#pragma once

#include "NumHalide/NumHalide.h"

NS_NUM_HALIDE_FUNC_BEGIN

template < typename ValueType >
bool is_const_value(Expr x, ValueType v)
{
	if (int64_t const* pValue = as_const_int(x))
	{
		return static_cast<ValueType>(*pValue) == v;
	}
	else if (uint64_t const* pValue = as_const_uint(x))
	{
		return static_cast<ValueType>(*pValue) == v;
	}
	else if (f64 const* pValue = as_const_float(x))
	{
		return static_cast<ValueType>(*pValue) == v;
	}
	return is_const(x, static_cast<Int64>(v));
}

inline
Func	linspace( Type type, Expr _start, Expr _stop, Expr num = 50, std::string const& name = "linspace", Bool endpoint = true, Int32 axis = 0)
{
	Func ret(name);
	Var x;

	Expr start = cast(type, _start);
	Expr stop = cast(type, _stop);

	require(start < stop, {start, stop});
	require(num > 0, { num });

	if ( is_const_zero(num) )
	{
		ret(x) = Internal::make_const( type, 0 );

		ret.bound_storage(x, 1);
	}

	if ( is_const_one(num) )
	{
		Func ret;

		ret(x) = start;
		ret.bound_storage(x, 1);
	}

	if ( endpoint )
	{
		if (is_const_value(num, 2))
		{
			Func ret;
			ret(x) = mux(x, { start, stop });
			ret.specialize(start < stop);
			ret.bound_storage(x, 2);
		}
		else
		{
			Expr step = ( stop - start ) / cast( type, num - 1 );

			ret(x) = start + cast(type, x) * step;
			ret.specialize(start < stop);
			ret.bound_storage(x, num);
		}
	}
	else
	{
		if (is_const_value(num, 2))
		{
			Expr step = ( stop - start ) / cast(type, num);
			ret(x) = mux(x, { start, start + step });
			ret.specialize(start < stop);
			ret.bound_storage(x, 2);
		}
		else
		{
			Expr step = ( stop - start ) / cast(type, num);

			ret(x) = start + cast(type, x) * step;
			ret.specialize(start < stop);
			ret.bound_storage(x, num);
		}
	}

	if ( axis > 0 )
	{
		NH_ASSERT(false);
	}

	return ret;
}

inline
Func	arange(Type type, Expr _stop, std::string const& name = "arange")
{
	Func ret(name);
	Var x;

	Expr stop = cast(type, _stop);

	require(stop > 0, stop);

	ret(x) = cast(type, x);
	ret.specialize(stop > Internal::make_const(type, 0));
	ret.bound_storage(x, stop);

	return ret;
}

inline
Func	arange(Type type, Expr _start, Expr _stop, Expr _step = 1, std::string const& name = "arange")
{
	Func ret(name);
	Var x;

	Expr start = cast(type, _start);
	Expr stop = cast(type, _stop);
	Expr step = cast(type, _step);

	require(( step > Internal::make_const(type, 0) && stop > start )
		||
			( step < Internal::make_const(type, 0) && start > stop ),
		{start, stop, step}
	);

	ret(x) = cast(type, start + step*cast(type, x));

	ret.bound_storage(x,
					  select(step > Internal::make_const(type, 0),
							 ( stop - start ) / step,
							 ( start - stop ) / step
						  ) );

	return ret;
}

inline
Func	full( Type type, Expr value, Int32 dim, std::string const& name = "full")
{
	Func ret(name);
	std::vector<Var> vars;
	vars.resize(dim);

	ret(vars) = cast(type, value);

	return ret;
}

inline
Func	empty( Type type, Int32 dim, std::string const& name = "empty")
{
	Func ret(name);
	std::vector<Var> vars;
	vars.resize(dim);

	ret(vars) = undef(type);

	return ret;
}

inline
Func	ones( Type type, Int32 const dim, std::string const& name = "one")
{
	return full(type, 1, dim, name);
}

inline
Func	zeros( Type type, Int32 const dim, std::string const& name = "zeros")
{
	return full(type, 0, dim, name);
}

inline
Func	diag(Type type, Expr val, Int32 const dim, std::string const& name = "diag")
{
	Func ret(name);
	std::vector<Var> vars;
	vars.resize(dim);

	ret(vars) = cast(type, 0);
	std::vector<Var> args(dim, vars[0]);
	ret(args) = cast(type, val);

	return ret;
}

inline
Func	diag(Type type, Func values, Expr const k = 0, std::string const& name = "diag")
{
	Func ret(name);
	Var x;
	Var y;

	if (is_const_zero(k))
	{
		ret(x, y) = cast(type, 0);
		ret(x, x) = cast(type, values(x));
	}
	else
	{
		ret(x, y) = select(x - k == y, cast(type, values(x)), cast(type, 0));
	}

	return ret;
}

inline
Func	diag(Type type, Buffer<> const& values, Expr const k = 0, std::string const& name = "diag")
{
	Func ret(name);
	Var x;
	Var y;

	if (is_const_zero(k))
	{
		ret(x, y) = cast(type, 0);
		ret(x, x) = cast(type, values(x));
	}
	else
	{
		ret(x, y) = select(x - k == y, cast(type, values(x)), cast(type, 0));
	}

	return ret;
}

inline
Func	identity(Type type, Int32 const dim, std::string const& name = "identity")
{
	return diag(type, 1, dim, name);
}

inline
std::vector<Func>	meshgrid(Type type, std::vector<Func> xis, std::string const& name = "meshgrid")
{
	Int32 rank = (Int32)xis.size();

	std::vector<Func> values;
	values.reserve(rank);
	for (Int32 k = 0; k < rank; ++k)
	{
		Func base = empty(type, rank, name + '_' + std::to_string(k));
		std::vector<Var> const& vars = base.args();
		base(vars) = xis[k](vars[k]);

		values.push_back(base);
	}

	return values;
}

NS_NUM_HALIDE_FUNC_END
