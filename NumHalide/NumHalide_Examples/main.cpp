#include <iostream>

#include "NumHalide/NumHalide.h"
#include "NumHalide/Common.h"
//#include "NumHalide/NDArray.h"
#include "NumHalide/InitializersFunc.h"
#include "NumHalide/ManipulationFunc.h"
//#include "NumHalide/InitializersArr.h"

//#pragma comment( lib, "NumHalide.lib" )
#pragma comment( lib, "Halide.lib" )

#define NUMCPP_NO_USE_BOOST
#include "NumCpp.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <cstdlib>

using namespace numhalide;
using namespace numhalide::func;
//using namespace numhalide::arr;
using namespace Halide;

Func ImgFunc()
{
	int width = 256;
	Func xs = func::linspace(Float(32), 0.0f, 1.0f, width, "xs");
	Func ys = func::linspace(Float(32), 0.0f, 1.0f, width, "ys");
	std::vector<Func> ids = numhalide::func::meshgrid(Float(32), { xs, ys }, "meshgrid");
	Func x = ids[0];
	Func y = ids[1];

	Var u, v, c;

	Expr cx = 2.0f * (x(u, v) - 0.5f);
	Expr cy = 2.0f * (y(u, v) - 0.5f);

	Expr out = exp(-(cx * cx + cy * cy) / 0.25f);

	Expr to8bits = cast(UInt(8), round(pow(clamp(out, 0.0f, 1.0f), 1.0f / 2.2f) * 255.0f));

	Func result(UInt(8), 3, "image");
	result(c, u, v) = select(c < 3, to8bits, Internal::make_const(UInt(8), 255));
	//result(0, u, v) = to8bits;
	//result(1, u, v) = to8bits;
	//result(2, u, v) = to8bits;
	//result(4, u, v) = Internal::make_const(UInt(8), 255);
	result.set_estimates({ { 0, 4 }, { 0, width }, { 0, width } });
	xs.set_estimates({ { 0, width } });
	ys.set_estimates({ { 0, width } });

	return result;
}

int main()
{
	try
	{
#if 1
//#if 1
		{
			Int32 num = 5;
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
			Func id = numhalide::func::identity(Float(32), 2);
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
			Func id = numhalide::func::ones(Float(32), 2);
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
			Func id = numhalide::func::zeros(Float(32), 2);
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
			Func id = numhalide::func::full(Float(32), 3.14f, 2);
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
			std::vector<f32> vals = { 1.41f, 2.16f, 3.14f };
			Buffer<f32> bufValues(&vals[0], { 3 }, "values");
			Func id = numhalide::func::diag(Float(32), bufValues );
			Buffer<f32> bId = id.realize({ 3, 3 });

			std::cout << "func::diag(Float(32), (buffers){ 1.41f, 2.16f, 3.14f });" << std::endl;
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
			std::vector<f32> vals = { 1.41f, 2.16f, 3.14f };
			Buffer<f32> bufValues(&vals[0], { 3 }, "values");
			Func id = numhalide::func::diag(Float(32), bufValues, 1 );
			Buffer<f32> bId = id.realize({ 3, 3 });

			std::cout << "func::diag(Float(32), (buffers){ 1.41f, 2.16f, 3.14f }, 1);" << std::endl;
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
			std::vector<f32> vals = { 1.41f, 2.16f, 3.14f };
			Buffer<f32> bufValues(&vals[0], { 3 }, "values");
			Func values;
			Var x;
			values(x) = bufValues(x);
			Func id = numhalide::func::diag(Float(32), values );
			Buffer<f32> bId = id.realize({ 3, 3 });

			std::cout << "func::diag(Float(32), (func){ 1.41f, 2.16f, 3.14f });" << std::endl;
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
			std::vector<f32> vals = { 1.41f, 2.16f, 3.14f };
			Buffer<f32> bufValues(&vals[0], { 3 }, "values");
			Func values;
			Var x;
			values(x) = bufValues(x);
			Func id = numhalide::func::diag(Float(32), values, -1 );
			Buffer<f32> bId = id.realize({ 3, 3 });

			std::cout << "func::diag(Float(32), (func){ 1.41f, 2.16f, 3.14f }, -1);" << std::endl;
			for (int j = 0; j < 3; ++j)
			{
				for (int i = 0; i < 3; ++i)
				{
					std::cout << bId(i, j) << ' ';
				}
				std::cout << std::endl;
			}
		}
#endif
//#elif 1
#if 1
		{
			//Int32 count_x_val = 16;
			//Int32 count_y_val = 16;
			//Buffer<Int32> count_x = Buffer<Int32>::make_scalar(&count_x_val, "count_x");
			//Buffer<Int32> count_y = Buffer<Int32>::make_scalar(&count_y_val, "count_y");
			//Expr xx = count_x();
			//Expr yy = count_y();
			Int32 height0 = 8;
			Int32 height1 = 4;
			Param<int> xx;
			xx.set(height0);
			xx.set_estimate(height0);
			Param<int> yy;
			xx.set(height1);
			xx.set_estimate(height1);
			Func xs = func::linspace(Float(32), 0.0f, 1.0f, xx, "xs");
			Func ys = func::linspace(Float(32), 0.0f, 1.0f, yy, "ys");
			std::vector<Func> ids = numhalide::func::meshgrid(Float(32), { xs, ys }, "meshgrid");
			for (Int32 k = 0; k < 2; ++k)
			{
				ids[k].compile_to_c(std::to_string(k) + ".cpp",
					{ xx, yy }, "_" + std::to_string(k));
				ids[k].compile_to_lowered_stmt(std::to_string(k) + ".html",
					{ xx, yy },
					StmtOutputFormat::HTML);
			}
			Buffer<f32> results[2];
			results[0] = Buffer<f32>(Float(32), { height0, height1 });
			results[1] = Buffer<f32>(Float(32), { height0, height1 });
			for (Int32 k = 0; k < 2; ++k)
			{
				Callable call = ids[k].compile_to_callable({ xx, yy });
				//results[k] = ids[k].realize({ 8, 8 }, Target(), pm);
				//results[k] = call(&empty, &height0, &height1);
				call(height0, height1, results[k]);
			}
			std::cout << std::endl;
			std::cout << std::endl;
			for (Int32 k = 0; k < 2; ++k)
			{
				ids[k].print_loop_nest();
				std::cout << std::endl;
			}
			std::cout << std::endl;
			std::cout << std::endl;
			std::cout << "xs = func::linspace(Float(32), 0.0f, 1.0f, 16);" << std::endl;
			std::cout << "ys = func::linspace(Float(32), 0.0f, 1.0f, 16);" << std::endl;
			std::cout << "func::meshgrid(Float(32), {xs, ys});" << std::endl;
			for (int j = 0; j < height1; ++j)
			{
				for (int i = 0; i < height0; ++i)
				{
					std::cout << results[0](i, j) << ' ';
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
			for (int j = 0; j < height1; ++j)
			{
				for (int i = 0; i < height0; ++i)
				{
					std::cout << results[1](i, j) << ' ';
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
//#endif
#endif

#if 1
		Buffer<f32> values(Float(32), { 5, 4 });
		values.for_each_element(
			[&](int x, int y)
			{
				values(x, y) = (f32)(5*y + x);
			});
		values.for_each_element(
			[&](int x, int y)
			{
				std::cout << values(x, y) << ' ';
				if (x == 4)
					std::cout << std::endl;
			});
		Var x, y;
		Func in;
		in(x, y) = values(x, y);

		//Func flat = func::flatten(in, {5, 4});
		//
		//Buffer< f32 > out( 20 );
		//out = flat.realize({ 20 });

		//std::cout << std::endl;
		//std::cout << std::endl;
		//out.for_each_element(
		//	[&](int x)
		//	{
		//		std::cout << out(x) << ' ';
		//	});

		Func res = reshape(in, { 5, 4 }, { 2, 10 });
		Buffer< f32 > outR(20);
		outR = res.realize({ 2, 10 });
		outR.for_each_element(
			[&](int x, int y)
			{
				std::cout << outR(x, y) << ' ';
				if (x == 1)
					std::cout << std::endl;
			});

		//res.compile_to(
		//	{
		//		{OutputFileType::cpp_stub, "fn.cpp"},
		//		{OutputFileType::device_code, "fn.device.c"},
		//		{OutputFileType::c_source, "fn.c"},
		//		//{OutputFileType::compiler_log, "fn.halide_compiler_log"},
		//		{OutputFileType::llvm_assembly, "fn.ll"},
		//		{OutputFileType::python_extension, "fn.py.cpp"},
		//		{OutputFileType::pytorch_wrapper, "fn.pytorch.h"},
		//		{OutputFileType::assembly, "fn.asm"},
		//		{OutputFileType::device_code, "fn.dc"},
		//		{OutputFileType::conceptual_stmt_html, "fn.html"},
		//		{OutputFileType::bitcode, "fn.bc"}
		//	}, {}, "");
#endif

#if 1
		{
			_putenv("HL_DEBUG_CODEGEN=1");
			//std::cout << "VarValue: " << Halide::Internal::get_env_variable("HL_DEBUG_CODEGEN") << std::endl;
			load_plugin("autoschedule_li2018.dll");
			load_plugin("autoschedule_adams2019.dll");
			load_plugin("autoschedule_anderson2021.dll");
			int width = 512;
			// CPU
			{
				Target t = get_target_from_environment()
					.with_feature(Target::Feature::NoBoundsQuery)
					.with_feature(Target::Feature::NoAsserts);

				Pipeline pCPU(ImgFunc());
				pCPU.apply_autoscheduler(t, { "Adams2019", {} });

				pCPU.print_loop_nest();
				pCPU.compile_to_c("image_cpu.cpp", {}, "image", t);
				pCPU.compile_to_lowered_stmt("image_cpu.html", {}, StmtOutputFormat::HTML);
			}

			/*
			// GPU
			{
				Target t = get_target_from_environment()
					.with_feature(Target::Feature::NoBoundsQuery)
					.with_feature(Target::Feature::NoAsserts)
					.with_feature(Target::Feature::CUDA);

				Pipeline pGPU(ImgFunc());
				pGPU.apply_autoscheduler(t, { "Anderson2021", {
						{ "num_passes", "10" },
						{ "search_space_options", "1111" },
						{ "shared_memory_limit_kb", "64" },
						{ "shared_memory_sm_limit_kb", "64" }
					} });

				pGPU.print_loop_nest();
				pGPU.compile_to_c("image_gpu.cpp", {}, "image", t);
				pGPU.compile_to_lowered_stmt("image_gpu.html", {}, StmtOutputFormat::HTML);
			}
			*/

			std::vector<UInt8> results;
			results.resize(width * width * 4);

			Callable call = ImgFunc().compile_to_callable({});

			Buffer<UInt8> resBuf(UInt(8), &results[0], { 4, width, width });

			call(resBuf);

			stbi_write_png("img.png", width, width, 4, &results[0], 0);
		}
#endif
	}
	catch ( Error& e )
	{
		std::cout << e.what() << std::endl;
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

	return 0;
}
