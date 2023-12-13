#pragma once

#include "NumHalide/NumHalide.h"

NS_NUM_HALIDE_ARRAY_BEGIN

template < typename ValueType >
inline
Buffer<ValueType>	linspace( Type const type, ValueType const start, ValueType const stop, Int32 const num = 50, Bool const endpoint = true, Int32 const axis = 0 )
{
	Buffer<ValueType> ret;

	NH_ASSERT(num >= 0);
	NH_ASSERT(start < stop);

	if ( num == 0 )
	{
		ret = Buffer< ValueType >(type, { 1 });
		ret(0) = static_cast< ValueType >( 0 );
	}

	if ( num == 1 )
	{
		ret = Buffer< ValueType >(type, { 1 });
		ret(0) = start;
	}

	if ( endpoint )
	{
		if ( num == 2 )
		{
			ret = Buffer< ValueType >(type, { 2 });
			ret(0) = start;
			ret(1) = stop;
		}
		else
		{
			ValueType const step = ( stop - start ) / static_cast< ValueType >( num - 1 );
			for ( Int32 i = 0; i < num - 1; ++i )
			{
				ret( i ) = start + static_cast< ValueType >( i ) * step;
			}
		}
	}
	else
	{
		ValueType const step = ( stop - start ) / static_cast< ValueType >( num );
		if ( num == 2 )
		{
			ret = Buffer< ValueType >(type, { 2 });
			ret(0) = start;
			ret(1) = start + step;
		}
		else
		{
			for ( Int32 i = 0; i < num - 1; ++i )
			{
				ret(i) = start + static_cast< ValueType >( i ) * step;
			}
		}
	}

	if ( axis > 0 )
	{
		for ( Int32 i = 0; i < axis; ++i )
		{
			ret.add_dimension();
		}
		ret.transpose(0, axis);
	}

	return ret;
}

template < typename ValueType >
inline
Buffer<ValueType>	arange(Type const type, ValueType const start, ValueType const stop, ValueType const step = static_cast< ValueType >( 1 ) )
{
	Buffer<ValueType> ret;

	NH_ASSERT(	( step > Internal::make_const(type, 0) && stop > start )
			||
				( step < Internal::make_const(type, 0) && start > stop )
		);

	ValueType value = start;
	ValueType counter = static_cast< ValueType >( 1 );

	Int32 k = 0;
	if ( step > static_cast< ValueType >( 0 ) )
	{
		while ( value < stop )
		{
			ret(k++) = value;
			value = start + step * counter++;
		}
	}
	else
	{
		while ( value > stop )
		{
			ret(k++) = value;
			value = start + step * counter++;
		}
	}

	return ret;
}

template < typename ValueType >
inline
Buffer<ValueType>	arange(Type const type, ValueType const stop)
{
	NH_ASSERT(stop > static_cast< ValueType >( 0 ));

	return arange< ValueType >( type, static_cast< ValueType >( 1 ), stop, static_cast< ValueType >( 1 ) );
}

template < typename ValueType >
inline
Buffer<ValueType>	full( Type const type, ValueType const value, Int32 const size )
{
	Buffer<ValueType> ret = Buffer< ValueType >(type, { size });

	ret.fill(value);

	return ret;
}

template < typename ValueType >
inline
Buffer<ValueType>	full( Type const type, ValueType const value, Int32 const rows, Int32 const cols )
{
	Buffer<ValueType> ret = Buffer< ValueType >(type, { rows, cols });

	ret.fill(value);

	return ret;
}

template < typename ValueType >
inline
Buffer<ValueType>	full( Type const type, ValueType const value, std::vector<Int32> const& sizes )
{
	Buffer<ValueType> ret = Buffer< ValueType >(type, sizes);

	ret.fill(value);

	return ret;
}

template < typename ValueType >
inline
Buffer<ValueType>	empty( Type const type, std::vector<Int32> const& sizes )
{
	return Buffer< ValueType >( type, sizes );
}

template < typename ValueType >
inline
Buffer<ValueType>	ones( Type const type, Int32 const size )
{
	return full(type, 1, size);
}

template < typename ValueType >
inline
Buffer<ValueType>	ones( Type const type, Int32 const rows, Int32 const cols )
{
	return full(type, 1, rows, cols);
}

template < typename ValueType >
inline
Buffer<ValueType>	ones( Type const type, std::vector<Int32> const& sizes )
{
	return full(type, 1, sizes);
}

template < typename ValueType >
inline
Buffer<ValueType>	zeros( Type const type, Int32 const size )
{
	return full(type, 0, size);
}

template < typename ValueType >
inline
Buffer<ValueType>	zeros( Type const type, Int32 const rows, Int32 const cols )
{
	return full(type, 0, rows, cols);
}

template < typename ValueType >
inline
Buffer<ValueType>	zeros( Type const type, std::vector<Int32> const& sizes )
{
	return full(type, 0, sizes);
}

template < typename ValueType >
inline
Buffer<ValueType>	identity( Type const type, Int32 const side )
{
	Buffer<ValueType> ret = zeros<ValueType>( type, side, side );

	for ( Int32 i = 0; i < side; ++i )
	{
		ret(i, i) = 1;
	}

	return ret;
}

template < typename ValueType >
inline
Buffer<ValueType>	diag( Type const type, Buffer<ValueType> const& values, Int32 const side )
{
	Buffer<ValueType> ret = zeros<ValueType>( type, side, side );

	for ( Int32 i = 0; i < side; ++i )
	{
		ret(i, i) = values(i);
	}

	return ret;
}

template < typename ValueType >
inline
Buffer<ValueType>	diag( Type const type, Buffer<ValueType> const& values, Int32 const side, Int32 const k = 0 )
{
	Buffer<ValueType> ret = zeros<ValueType>( type, side, side );

	for ( Int32 i = 0; i < side; ++i )
	{
		ret(i + k, i + k) = values(i);
	}

	return ret;
}

NS_NUM_HALIDE_ARRAY_END
