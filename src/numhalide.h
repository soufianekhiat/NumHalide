#pragma once

#include "IncludeHalide.h"

// Simplified namespace - we only support Func-based operations
#define NS_NUM_HALIDE_BEGIN	namespace numhalide {
#define NS_NUM_HALIDE_END	}

NS_NUM_HALIDE_BEGIN

// Default to 32-bit scalars if not specified
#ifndef NS_SCALAR_BITS
#define NS_SCALAR_BITS 32
#endif

#define NH_ASSERT( X ) assert( X )

// Basic type aliases
typedef	bool					Bool;

typedef	uint8_t					UInt8;
typedef	uint16_t				UInt16;
typedef	uint32_t				UInt32;
typedef	uint64_t				UInt64;

typedef	int8_t					Int8;
typedef	int16_t					Int16;
typedef	int32_t					Int32;
typedef	int64_t					Int64;

typedef	void*					Ptr;
typedef	void const*				PtrConst;
typedef	void const* const		PtrConst_Const;
typedef	void* const				Ptr_Const;

typedef	float					f32;
typedef	double					f64;
typedef	long double				f80;

// Scalar type based on NS_SCALAR_BITS
#if NS_SCALAR_BITS == 16
typedef	f32						Scalar;
#elif NS_SCALAR_BITS == 32
typedef	f32						Scalar;
#elif NS_SCALAR_BITS == 64
typedef	f64						Scalar;
#else
#error "NS_SCALAR_BITS must be 16, 32, or 64"
#endif

typedef	decltype( nullptr )		NullPtrType;

NS_NUM_HALIDE_END
