#pragma once

#include "NumHalide/NumHalide.h"

NS_NUM_HALIDE_FUNC_BEGIN

inline
Func	linspace( Type type, Expr _start, Expr _stop, Int32 num = 50, Bool endpoint = true, Int32 axis = 0 )
{
	Func ret;
	Var x;

	Expr start = cast(type, _start);
	Expr stop = cast(type, _stop);

	NH_ASSERT(num >= 0);

	require(start < stop, {start, stop});

	if ( num == 0 )
	{
		ret(x) = Internal::make_const( type, 0 );

		ret.bound_storage(x, 1);
	}

	if ( num == 1 )
	{
		Func ret;

		ret(x) = start;
		ret.bound_storage(x, 1);
	}

	if ( endpoint )
	{
		if ( num == 2 )
		{
			Func ret;
			ret(x) = mux(x, { start, stop });
			ret.specialize(start < stop);
			ret.bound_storage(x, 2);
		}
		else
		{
			Expr step = ( stop - start ) / Internal::make_const( type, num - 1 );

			ret(x) = start + cast(type, x) * step;
			ret.specialize(start < stop);
			ret.bound_storage(x, num);
		}
	}
	else
	{
		if ( num == 2 )
		{
			Expr	step = ( stop - start ) / Internal::make_const(type, num);
			ret(x) = mux(x, { start, start + step });
			ret.specialize(start < stop);
			ret.bound_storage(x, 2);
		}
		else
		{
			Expr step = ( stop - start ) / Internal::make_const(type, num);

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
Func	arange(Type type, Expr _stop)
{
	Func ret;
	Var x;

	Expr stop = cast(type, _stop);

	require(stop > 0, stop);

	ret(x) = cast(type, x);
	ret.specialize(stop > Internal::make_const(type, 0));
	ret.bound_storage(x, stop);

	return ret;
}

inline
Func	arange(Type type, Expr _start, Expr _stop, Expr _step = 1)
{
	Func ret;
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
Func	full( Type type, Expr value, Int32 const size )
{
	Func ret;
	Var x;

	ret(x) = cast(type, value);

	return ret;
}

inline
Func	full( Type type, Expr value, Int32 const rows, Int32 const cols )
{
	Func ret;
	Var x;
	Var y;

	ret(x, y) = cast(type, value);

	return ret;
}

inline
Func	full( Type type, Expr value, std::vector<Int32> const& sizes )
{
	Func ret;
	std::vector<Var> vars{sizes.size()};

	ret(vars) = cast(type, value);

	return ret;
}

inline
Func	ones( Type type, Int32 const size )
{
	return full(type, 1, size);
}

inline
Func	ones( Type type, Int32 const rows, Int32 const cols )
{
	return full(type, 1, rows, cols);
}

inline
Func	ones( Type type, std::vector<Int32> const& sizes )
{
	return full(type, 1, sizes);
}

inline
Func	zeros( Type type, Int32 const size )
{
	return full(type, 0, size);
}

inline
Func	zeros( Type type, Int32 const rows, Int32 const cols )
{
	return full(type, 0, rows, cols);
}

inline
Func	zeros( Type type, std::vector<Int32> const& sizes )
{
	return full(type, 0, sizes);
}

inline
Func	identity(Type type, Int32 const side)
{
	Func ret;
	Var x;
	Var y;

	ret(x, y) = cast(type, 0);
	ret(x, x) = cast(type, 1);

	return ret;
}

NS_NUM_HALIDE_FUNC_END
