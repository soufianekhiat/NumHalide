#pragma once

#include "NumHalide/NumHalide.h"
#include "NumHalide/Common.h"

#include <vector>
#include <tuple>
#include <numeric>

NS_NUM_HALIDE_ARRAY_BEGIN

enum class Order : UInt8
{
	RowMajor,
	ColMajor,
	C = RowMajor,
	F = ColMajor
};

template < typename Type >
class NDArray : public Buffer< Type >
{
public:
	NDArray() = delete;
	NDArray( Type const type, Int32 dim )
	:	Buffer<>( dim )
	,	m_dimensions{ dim }
	,	m_vars		{ Var{} }
	,	m_order		{ Order::ColMajor }
	{}

	NDArray( Type const type, std::vector< Int32 > const& e, Order const order = Order::ColMajor )
	:	Buffer<>( e )
	,	m_dimensions( e )
	,	m_vars		( e.size() )
	,	m_order		{ order }
	{}

	static
	NDArray< Type >	from_data( std::initializer_list< Type >&& in )
	{
		NDArray< Type > newArray( ( Int32 )in.size() );
		*reinterpret_cast< Buffer< Type >* >( &newArray ) = Buffer< Type >( ( Int32 )in.size() );
		newArray.m_dimensions	= { 1 };
		newArray.m_vars			= { Var{} };
		newArray.m_order		= Order::ColMajor;

		return newArray;
	}

	NDArray( NDArray< Type > const& others )
	:	Buffer< Type >( others.to_buffer() )
	,	m_dimensions( others.m_dimensions )
	,	m_vars		{ others.m_vars }
	,	m_order		{ others.m_order }
	{}

	NDArray const& operator=( NDArray< Type > const& others )
	{
		*reinterpret_cast< Buffer< Type >* >( this ) = others.to_buffer();
		m_dimensions	= others.m_dimensions;
		m_vars			= others.m_vars;
		m_order			= others.m_order;
	}

	template < typename... Ints >
	Type&	operator()( Ints... arguments )
	{
		typedef typename std::tuple_element< 0, std::tuple< Ints... > >::type FirstType;

		NH_ASSERT( sizeof...( arguments ) == m_dimensions.size() );

		Buffer< Type >::set_host_dirty();

		if ( m_order == Order::ColMajor )
			return Buffer< Type >::operator()( std::forward< Ints >( arguments )... );
		else
			return apply_reverse< Type& >( *reinterpret_cast< Buffer< Type >* >( this ), std::forward< Ints >( arguments )... );
	}

	Buffer< Type >&			to_buffer()
	{
		return static_cast< Buffer< Type >& >( *this );
	}
	Buffer< Type > const&	to_buffer() const
	{
		return static_cast< Buffer< Type > const& >( *this );
	}

	Order	order() const
	{
		return m_order;
	}

	void	set_order( Order const order )
	{
		m_order = order;
	}

	Type*	begin()
	{
		return Buffer< Type >::begin();
	}
	Type*	end()
	{
		return Buffer< Type >::end();
	}

	Type&	front()
	{
		return *Buffer< Type >::begin();
	}
	Type&	back()
	{
		return *( Buffer< Type >::data() + std::accumulate( m_dimensions.begin(), m_dimensions.end(), 1, std::multiplies< Int32 >() ) - 1 );
	}

	Func	func() const
	{
		Func f;

		std::vector< Expr > exprs( m_vars.size() );
		std::transform( m_vars.begin(), m_vars.end(), exprs.begin(),
					   []( Var const e )
					   {
						   return ( Expr )e;
					   } );
		f( m_vars ) = ( *this )( exprs );

		return f;
	}

	Func	flat_func() const
	{
		Func f;

		std::vector< Expr > exprs( m_vars.size() );
		std::transform( m_vars.begin(), m_vars.end(), exprs.begin(),
					   []( Var const e )
					   {
						   return ( Expr )e;
					   } );
		std::vector< Expr > dims( m_dimensions.size() );
		std::transform( m_dimensions.begin(), m_dimensions.end(), dims.begin(),
					   []( Int32 const d )
					   {
						   return ( Expr )d;
					   });
		Var k;
		std::vector< Expr > args;
		index_to_args< Expr >( args, k, m_dimensions );
		f( k ) = ( *this )( args );

		return f;
	}

	std::vector< Int32 >	shape() const
	{
		return m_dimensions;
	}

	template < typename... Ints >
	void	reshape( Ints... arguments )
	{
		NH_ASSERT( tproduct( arguments... ) == std::accumulate( m_dimensions.begin(), m_dimensions.end(), 1, std::multiplies< Int32 >() ) );

		m_dimensions.clear();
		m_dimensions = { arguments... };
		m_vars.resize( m_dimensions.size() );
		*reinterpret_cast< Buffer< Type >* >( this ) = Buffer< Type >( Buffer< Type >::data(), m_dimensions );
		Buffer< Type >::set_host_dirty();
	}

	template < typename IntType >
	void	reshape( std::vector< IntType > const& arguments )
	{
		if constexpr ( std::is_same_v< IntType, Int32 > )
		{
			NH_ASSERT(std::accumulate(arguments.begin(), arguments.end(), 1, std::multiplies< IntType >()) ==
					  std::accumulate(m_dimensions.begin(), m_dimensions.end(), 1, std::multiplies< Int32 >()));
		}

		m_dimensions.clear();
		//m_dimensions = { arguments.rbegin(), arguments.rend() };
		m_dimensions = arguments;
		m_vars.resize( m_dimensions.size() );
		*reinterpret_cast< Buffer< Type >* >( this ) = Buffer< Type >( Buffer< Type >::data(), m_dimensions );
		Buffer< Type >::set_host_dirty();
	}

	void	fill( Type const x )
	{
		Buffer< Type >::fill( x );
	}

	void	zeros()
	{
		fill( static_cast< Type >( 0 ) );
	}

	void	ones()
	{
		fill( static_cast< Type >( 1 ) );
	}

private:
	Type					m_type;
	std::vector< Int32 >	m_dimensions;
	std::vector< Var >	m_vars;
	Order					m_order;
};

NS_NUM_HALIDE_ARRAY_END
