#include <iostream>

#include "NumHalide/NumHalide.h"
#include "NumHalide/Common.h"
//#include "NumHalide/NDArray.h"
#include "NumHalide/InitializersFunc.h"
#include "NumHalide/InitializersArr.h"

//#pragma comment( lib, "NumHalide.lib" )
#pragma comment( lib, "Halide.lib" )

#define NUMCPP_NO_USE_BOOST
#include "NumCpp.hpp"

using namespace numhalide;
using namespace numhalide::func;
using namespace numhalide::arr;
using namespace Halide;

int main()
{
	try
	{
#if 0
		int num = 5;
		Func ls0 = linspace(Float(32), 2.0f, 3.0f, num);

		Buffer< f32 > r0(num);
		r0 = ls0.realize({ num });

		for ( int k = 0; k < num; ++k )
		{
			std::cout << r0(k) << ' ';
		}
		std::cout << std::endl;

		Func ls1 = linspace(Float(32), 2.0f, 3.0f, num, false);

		Buffer< f32 > r1(num);
		r1 = ls1.realize({ num });

		for ( int k = 0; k < num; ++k )
		{
			std::cout << r1(k) << ' ';
		}
		std::cout << std::endl;
#elif 0
		int num = 20;
		Func ls0 = arange(Float(32), num);

		Buffer< f32 > r0(num);
		r0 = ls0.realize({ num });

		for ( int k = 0; k < num; ++k )
		{
			std::cout << r0(k) << ' ';
		}
		std::cout << std::endl;

		Func ls1 = arange(Int(32), 0.0f, 5.0f, 0.5f);

		num = 20;
		Buffer< Int32 > r1(num);
		r1 = ls1.realize({ num });

		for ( int k = 0; k < num; ++k )
		{
			std::cout << r1(k) << ' ';
		}
		std::cout << std::endl;
#elif 1
		Func id = identity(Float(32), 3);
		Buffer<f32> bId = id.realize({3, 3});

		for ( int j = 0; j < 3; ++j )
		{
			for ( int i = 0; i < 3; ++i )
			{
				std::cout << bId(i, j) << ' ';
			}
			std::cout << std::endl;
		}
#endif
	}
	catch ( std::exception& e )
	{
		std::cout << e.what() << std::endl;
	}

	return 0;
}
