#pragma once

#include "NumHalide/NumHalide.h"

NS_NUM_HALIDE_BEGIN

template < typename Type, typename SizeType >
inline
Type	args_to_index(	std::vector< Type > const& aArgs,
						std::vector< SizeType > const& aSizes )
{
	NH_ASSERT( aArgs.size() == aSizes.size() );

	int const iCount = static_cast< int >( aArgs.size() );

	NH_ASSERT( iCount >= 1 );

	if ( iCount == 1 )
	{
		return aArgs[ 0 ];
	}
	else
	{
		Type	uLast = aArgs[ iCount - 1 ]*aSizes[ iCount - 2 ] + aArgs[ iCount - 2 ];
		for ( int uIdx = 1; uIdx < iCount - 1; ++uIdx )
		{
			uLast = uLast*aSizes[ iCount - 2 - uIdx ] + aArgs[ iCount - 2 - uIdx ];
		}

		return uLast;
	}
}

template < typename Type, typename IndexType >
inline
void	index_to_args(	std::vector< Type >&		aArgs,
						IndexType const				iIndex,
						std::vector< Type > const&	aSizes )
{
	aArgs.clear();

	int const uCount = static_cast< int >( aSizes.size() );
	NH_ASSERT( uCount > 0 );

	Type oIndexCur = iIndex;

	auto oIt = aSizes.begin();

	for ( int iIdx = 0; iIdx < uCount; ++iIdx )
	{
		aArgs.push_back( oIndexCur % *oIt );
		oIndexCur = ( oIndexCur - aArgs.back() )/( *oIt );

		++oIt;
	}
}

template < typename FuncType, typename Type >
inline
Type	fold( FuncType F, Type first )
{
	return first;
}

template < typename FuncType, typename Type, typename... Args >
inline
Type	fold( FuncType F, Type first, Args... args )
{
	return F( first, fold( F, args... ) );
}

template < typename Type >
inline
Type	tsum(Type first)
{
	return first;
}

template < typename Type, typename... Args >
inline
Type	tsum(Type first, Args... args)
{
	return first + tsum(args...);
}

template < typename Type >
inline
Type	tproduct( Type first )
{
	return first;
}

template < typename Type, typename... Args >
inline
Type	tproduct( Type first, Args... args )
{
	return first*tproduct( args... );
}

template < typename Type >
inline
Type	tmin( Type a, Type b )
{
	return a < b ? a : b;
}

template < typename Type, typename... Args >
inline
Type	tmin( Type first, Args... args )
{
	return tmin( first, tmin( args... ) );
}

template < typename Type >
inline
Type	tmax( Type a, Type b )
{
	return a > b ? a : b;
}

template < typename Type, typename... Args >
inline
Type	tmax( Type first, Args... args )
{
	return tmax( first, tmax( args... ) );
}

template < typename RetType, typename FuncType, typename Type, typename... Args >
inline
RetType		apply_reverse( FuncType f, Type first, Args... args )
{
	return f( args..., first );
}

NS_NUM_HALIDE_END
