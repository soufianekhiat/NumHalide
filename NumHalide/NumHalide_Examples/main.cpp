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
//#if 1
		{
			int num = 5;
			Func ls0 = numhalide::func::linspace(Float(32), 2.0f, 3.0f, num);

			Buffer< f32 > r0(num);
			r0 = ls0.realize({ num });

			std::cout << "func::linspace(Float(32), 2.0f, 3.0f, num);" << std::endl;
			for (int k = 0; k < num; ++k)
			{
				std::cout << r0(k) << ' ';
			}
			std::cout << std::endl;

			Func ls1 = numhalide::func::linspace(Float(32), 2.0f, 3.0f, num, false);

			Buffer< f32 > r1(num);
			r1 = ls1.realize({ num });

			std::cout << "func::linspace(Float(32), 2.0f, 3.0f, num, false);" << std::endl;
			for (int k = 0; k < num; ++k)
			{
				std::cout << r1(k) << ' ';
			}
			std::cout << std::endl;
		}
//#elif 0
		{
			int num = 20;
			Func ls0 = numhalide::func::arange(Float(32), num);

			Buffer< f32 > r0(num);
			r0 = ls0.realize({ num });

			std::cout << "func::arange(Float(32), 20);" << std::endl;
			for (int k = 0; k < num; ++k)
			{
				std::cout << r0(k) << ' ';
			}
			std::cout << std::endl;

			Func ls1 = numhalide::func::arange(Int(32), 0.0f, 5.0f, 0.5f);

			num = 20;
			Buffer< Int32 > r1(num);
			r1 = ls1.realize({ num });

			std::cout << "func::arange(Int(32), 0.0f, 5.0f, 0.5f);" << std::endl;
			for (int k = 0; k < num; ++k)
			{
				std::cout << r1(k) << ' ';
			}
			std::cout << std::endl;

			Func ls2 = numhalide::func::arange(Float(32), 0.0f, 5.0f, 0.5f);

			num = 10;
			Buffer< f32 > r2(num);
			r2 = ls2.realize({ num });

			std::cout << "func::arange(Float(32), 0.0f, 5.0f, 0.5f);" << std::endl;
			for (int k = 0; k < num; ++k)
			{
				std::cout << r2(k) << ' ';
			}
			std::cout << std::endl;
		}
//#elif 1
		{
			Func id = numhalide::func::identity(Float(32), 5);
			Buffer<f32> bId = id.realize({ 5, 5 });

			std::cout << "func::identity(Float(32), 5);" << std::endl;
			for (int j = 0; j < 5; ++j)
			{
				for (int i = 0; i < 5; ++i)
				{
					std::cout << bId(i, j) << ' ';
				}
				std::cout << std::endl;
			}
		}
//#elif 1
		{
			Func id = numhalide::func::ones(Float(32), { 7, 7 });
			Buffer<f32> bId = id.realize({ 7, 7 });

			std::cout << "func::ones(Float(32), { 7, 7 });" << std::endl;
			for (int j = 0; j < 7; ++j)
			{
				for (int i = 0; i < 7; ++i)
				{
					std::cout << bId(i, j) << ' ';
				}
				std::cout << std::endl;
			}
		}
//#elif 1
		{
			Func id = numhalide::func::zeros(Float(32), { 3, 3 });
			Buffer<f32> bId = id.realize({ 3, 3 });

			std::cout << "func::zeros(Float(32), { 3, 3 });" << std::endl;
			for (int j = 0; j < 3; ++j)
			{
				for (int i = 0; i < 3; ++i)
				{
					std::cout << bId(i, j) << ' ';
				}
				std::cout << std::endl;
			}
		}
//#elif 1
		{
			Func id = numhalide::func::full(Float(32), 3.14f, { 3, 3 });
			Buffer<f32> bId = id.realize({ 3, 3 });

			std::cout << "func::full(Float(32), 3.14f, { 3, 3 });" << std::endl;
			for (int j = 0; j < 3; ++j)
			{
				for (int i = 0; i < 3; ++i)
				{
					std::cout << bId(i, j) << ' ';
				}
				std::cout << std::endl;
			}
		}
//#elif 1
		{
			std::vector<f32> vals = { 1.4142f, 2.16f, 3.14f };
			Buffer<f32> bufValues(&vals[0], { 3 }, "values");
			Func id = numhalide::func::diag(Float(32), bufValues, 3 );
			Buffer<f32> bId = id.realize({ 3, 3 });

			std::cout << "func::diag(Float(32), (buffers){ 1.4142f, 2.16f, 3.14f }, 3);" << std::endl;
			for (int j = 0; j < 3; ++j)
			{
				for (int i = 0; i < 3; ++i)
				{
					std::cout << bId(i, j) << ' ';
				}
				std::cout << std::endl;
			}
		}
//#elif 1
		{
			std::vector<f32> vals = { 1.4142f, 2.16f, 3.14f };
			Buffer<f32> bufValues(&vals[0], { 3 }, "values");
			Func values;
			Var x;
			values(x) = bufValues(x);
			Func id = numhalide::func::diag(Float(32), values, 3 );
			Buffer<f32> bId = id.realize({ 3, 3 });

			std::cout << "func::diag(Float(32), (func){ 1.4142f, 2.16f, 3.14f }, 3);" << std::endl;
			for (int j = 0; j < 3; ++j)
			{
				for (int i = 0; i < 3; ++i)
				{
					std::cout << bId(i, j) << ' ';
				}
				std::cout << std::endl;
			}
		}
//#endif
	}
	catch ( std::exception& e )
	{
		std::cout << e.what() << std::endl;
	}

	return 0;
}
