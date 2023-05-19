#pragma once

#include <Halide.h>

#define NS_NUM_HALIDE_BEGIN	namespace numhalide {
#define NS_NUM_HALIDE_END	}

#define NS_NUM_HALIDE_FUNC_BEGIN	namespace numhalide { namespace func {
#define NS_NUM_HALIDE_FUNC_END	} }

#define NS_NUM_HALIDE_ARRAY_BEGIN	namespace numhalide { namespace arr {
#define NS_NUM_HALIDE_ARRAY_END	} }

NS_NUM_HALIDE_BEGIN

//namespace H = Halide;
using namespace Halide;

#define NH_ASSERT( X ) assert( X )

typedef	bool					Bool;

typedef	uint8_t					UInt8;
typedef	uint16_t				UInt16;
typedef	uint32_t				UInt32;
typedef	uint64_t				UInt64;

typedef	int8_t					Int8;
typedef	int16_t					Int16;
typedef	int32_t					Int32;
typedef	int64_t					Int64;

typedef char					Char;
typedef wchar_t					WChar;
typedef char					Char;
typedef char					Char8;
typedef wchar_t					WChar;
typedef char16_t				Char16;
typedef char32_t				Char32;

typedef	char					CharASCII;
typedef	char					CharUTF8;
typedef	char16_t				CharUTF16;
typedef	char32_t				CharUTF32;
typedef	wchar_t					CharWide;

typedef	void*					Ptr;
typedef	void const*				PtrConst;
typedef	void const* const		PtrConst_Const;
typedef	void* const				Ptr_Const;

typedef	float					f32;
typedef	double					f64;
typedef	long double				f80;

#if SK_SCALAR_BITS == 32
typedef	f32						Scalar;
#else
typedef	f64						Scalar;
#endif

constexpr Scalar g_fDrawScale = static_cast<Scalar>(100.0);

#if SK_INTEGER_BITS == 32
typedef	Int32					Integer;
typedef	UInt32					UInteger;
#else
typedef	Int64					Integer;
typedef	UInt64					UInteger;
#endif

#ifdef __SK_WIN32__
typedef	UInt32					Address;
#elif defined( __SK_X64__ )
typedef	UInt64					Address;
#endif

typedef	decltype( nullptr )		NullPtrType;

#ifdef __SK_WIN32__
typedef	UInt16					UPlatformHalfWord;
typedef	Int16					PlatformHalfWord;
typedef	UInt32					UPlatformWord;
typedef	Int32					PlatformWord;
typedef	UInt64					UPlatformDoubleWord;
typedef	Int64					PlatformDoubleWord;
#elif defined( __SK_X64__ )
typedef	UInt32					UPlatformHalfWord;
typedef	Int32					PlatformHalfWord;
typedef	UInt64					UPlatformWord;
typedef	Int64					PlatformWord;
// Add UInt128 & SInt128 As {S}PlateformDoubleWord
#endif

NS_NUM_HALIDE_END
