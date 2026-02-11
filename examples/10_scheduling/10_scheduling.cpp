/// @file 10_scheduling.cpp
/// @brief Example 10: Scheduling helpers for performance optimization
///
/// Demonstrates:
///   - auto_tile() for cache-friendly tiling
///   - vectorize() for SIMD operations
///   - parallel() for multi-threading
///   - auto_schedule_2d() for automatic optimization
///   - Performance comparison between schedules
///
/// Output: out/10_scheduling_cpu.png

#include "numhalide_all.h"
#include "stbi_png.h"
#include <iostream>
#include <chrono>

using namespace numhalide;

// Helper to time a function
template<typename F>
double time_ms(F&& func) {
	auto start = std::chrono::high_resolution_clock::now();
	func();
	auto end = std::chrono::high_resolution_clock::now();
	return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char **argv) {
	try {
		const int size = 1024;

		std::cout << "Scheduling demonstration (" << size << "x" << size << " image)" << std::endl;
		std::cout << std::endl;

		// Create a computationally intensive function
		// Mandelbrot set computation
		Halide::Func mandelbrot("mandelbrot");
		Halide::Var x("x"), y("y");

		// Map pixel coordinates to complex plane [-2, 1] x [-1.5, 1.5]
		Halide::Expr cx = -2.0f + 3.0f * Halide::cast<float>(x) / size;
		Halide::Expr cy = -1.5f + 3.0f * Halide::cast<float>(y) / size;

		// Iterative computation (simplified for demo)
		const int max_iter = 50;
		Halide::Expr zx = 0.0f, zy = 0.0f;
		Halide::Expr iter = 0;

		// Unroll a few iterations for the demo
		for (int i = 0; i < max_iter; ++i) {
			Halide::Expr zx_new = zx * zx - zy * zy + cx;
			Halide::Expr zy_new = 2.0f * zx * zy + cy;
			Halide::Expr escaped = (zx * zx + zy * zy) > 4.0f;
			zx = Halide::select(escaped, zx, zx_new);
			zy = Halide::select(escaped, zy, zy_new);
			iter = Halide::select(escaped, iter, iter + 1);
		}

		mandelbrot(x, y) = Halide::cast<uint8_t>(iter * 255 / max_iter);

		// Version 1: No optimization (baseline)
		std::cout << "1. Baseline (no scheduling)..." << std::endl;
		Halide::Func baseline("baseline");
		baseline(x, y) = mandelbrot(x, y);

		Halide::Runtime::Buffer<uint8_t> out1(size, size);
		double time1 = time_ms([&]() { baseline.realize(out1); });
		std::cout << "   Time: " << time1 << " ms" << std::endl;

		// Version 2: Tiled
		std::cout << "2. Tiled (64x64)..." << std::endl;
		Halide::Func tiled("tiled");
		tiled(x, y) = mandelbrot(x, y);
		schedule::auto_tile(tiled, 64, 64);

		Halide::Runtime::Buffer<uint8_t> out2(size, size);
		double time2 = time_ms([&]() { tiled.realize(out2); });
		std::cout << "   Time: " << time2 << " ms" << std::endl;

		// Version 3: Vectorized
		std::cout << "3. Vectorized (width=" << schedule::get_vector_width() << ")..." << std::endl;
		Halide::Func vectorized("vectorized");
		vectorized(x, y) = mandelbrot(x, y);
		schedule::vectorize(vectorized, schedule::get_vector_width());

		Halide::Runtime::Buffer<uint8_t> out3(size, size);
		double time3 = time_ms([&]() { vectorized.realize(out3); });
		std::cout << "   Time: " << time3 << " ms" << std::endl;

		// Version 4: Full optimization (tiled + vectorized + parallel)
		std::cout << "4. Full optimization (tiled + vectorized + parallel)..." << std::endl;
		Halide::Func optimized("optimized");
		optimized(x, y) = mandelbrot(x, y);
		schedule::full_optimize_2d(optimized, 64, 64, schedule::get_vector_width());

		Halide::Runtime::Buffer<uint8_t> out4(size, size);
		double time4 = time_ms([&]() { optimized.realize(out4); });
		std::cout << "   Time: " << time4 << " ms" << std::endl;

		// Calculate speedups
		std::cout << std::endl;
		std::cout << "Speedups vs baseline:" << std::endl;
		std::cout << "   Tiled:     " << time1 / time2 << "x" << std::endl;
		std::cout << "   Vectorized:" << time1 / time3 << "x" << std::endl;
		std::cout << "   Full opt:  " << time1 / time4 << "x" << std::endl;

		// Save the optimized output (all versions produce identical Mandelbrot)
		const char* output_path = "out/10_scheduling_cpu.png";
		std::cout << std::endl << "Saving to " << output_path << "..." << std::endl;

		if (save_png(out4, output_path)) {
			std::cout << "Success! Mandelbrot (" << size << "x" << size << ") saved." << std::endl;
			std::cout << std::endl;
			std::cout << "Performance summary:" << std::endl;
			std::cout << "  Baseline:     " << time1 << " ms" << std::endl;
			std::cout << "  Tiled:        " << time2 << " ms (" << time1/time2 << "x faster)" << std::endl;
			std::cout << "  Vectorized:   " << time3 << " ms (" << time1/time3 << "x faster)" << std::endl;
			std::cout << "  Full opt:     " << time4 << " ms (" << time1/time4 << "x faster)" << std::endl;
		} else {
			std::cerr << "Error: Failed to save PNG" << std::endl;
			return 1;
		}

		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}
}
