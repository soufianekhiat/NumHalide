/// @file test_schedule.cpp
/// @brief Tests for scheduling helpers

#include <gtest/gtest.h>
#include "numhalide_all.h"

using namespace numhalide;

// -----------------------------------------------------------------------------
// Basic Scheduling Tests
// Note: These tests verify that scheduling operations don't break correctness
// -----------------------------------------------------------------------------

TEST(Schedule, AutoTile2D) {
	// Create a simple gradient
	Halide::Func f("tiled_func");
	Halide::Var x("x"), y("y");
	f(x, y) = Halide::cast<float>(x + y);

	// Apply tiling
	schedule::auto_tile(f, 16, 16);

	// Verify it still produces correct results
	Halide::Runtime::Buffer<float> out(64, 64);
	f.realize(out);

	for (int row = 0; row < 64; ++row) {
		for (int col = 0; col < 64; ++col) {
			float expected = static_cast<float>(col + row);
			EXPECT_NEAR(out(col, row), expected, 1e-5f);
		}
	}
}

TEST(Schedule, Vectorize) {
	// Create a simple gradient
	Halide::Func f("vectorized_func");
	Halide::Var x("x");
	f(x) = Halide::cast<float>(x * 2);

	// Apply vectorization
	schedule::vectorize(f, 8);

	// Verify correctness
	Halide::Runtime::Buffer<float> out(64);
	f.realize(out);

	for (int i = 0; i < 64; ++i) {
		EXPECT_NEAR(out(i), static_cast<float>(i * 2), 1e-5f);
	}
}

TEST(Schedule, Parallel) {
	// Create a 2D func
	Halide::Func f("parallel_func");
	Halide::Var x("x"), y("y");
	f(x, y) = Halide::cast<float>(x * y);

	// Apply parallelization on y
	schedule::parallel(f, 1);

	// Verify correctness
	Halide::Runtime::Buffer<float> out(32, 32);
	f.realize(out);

	for (int row = 0; row < 32; ++row) {
		for (int col = 0; col < 32; ++col) {
			float expected = static_cast<float>(col * row);
			EXPECT_NEAR(out(col, row), expected, 1e-5f);
		}
	}
}

TEST(Schedule, FullOptimize2D) {
	// Create a 2D func
	Halide::Func f("full_opt_func");
	Halide::Var x("x"), y("y");
	f(x, y) = Halide::cast<float>(x + y * 2);

	// Apply full optimization
	schedule::full_optimize_2d(f, 32, 32, 4);

	// Verify correctness
	Halide::Runtime::Buffer<float> out(128, 128);
	f.realize(out);

	for (int row = 0; row < 128; ++row) {
		for (int col = 0; col < 128; ++col) {
			float expected = static_cast<float>(col + row * 2);
			EXPECT_NEAR(out(col, row), expected, 1e-5f);
		}
	}
}

TEST(Schedule, Unroll) {
	// Create a simple func
	Halide::Func f("unrolled_func");
	Halide::Var x("x");
	f(x) = Halide::cast<float>(x * 3);

	// Apply unrolling
	schedule::unroll(f, 4);

	// Verify correctness
	Halide::Runtime::Buffer<float> out(32);
	f.realize(out);

	for (int i = 0; i < 32; ++i) {
		EXPECT_NEAR(out(i), static_cast<float>(i * 3), 1e-5f);
	}
}

TEST(Schedule, ComputeRoot) {
	// Create a chain of functions
	Halide::Func producer("producer");
	Halide::Func consumer("consumer");
	Halide::Var x("x"), y("y");

	producer(x, y) = Halide::cast<float>(x + y);
	consumer(x, y) = producer(x, y) * 2;

	// Force producer to compute at root
	schedule::compute_root(producer);

	// Verify correctness
	Halide::Runtime::Buffer<float> out(16, 16);
	consumer.realize(out);

	for (int row = 0; row < 16; ++row) {
		for (int col = 0; col < 16; ++col) {
			float expected = static_cast<float>((col + row) * 2);
			EXPECT_NEAR(out(col, row), expected, 1e-5f);
		}
	}
}

TEST(Schedule, GetVectorWidth) {
	// Just verify it returns a reasonable value
	int width = schedule::get_vector_width();
	EXPECT_GE(width, 4);
	EXPECT_LE(width, 16);
}

TEST(Schedule, AutoSchedule2D) {
	// Create a 2D func
	Halide::Func f("auto_scheduled_func");
	Halide::Var x("x"), y("y");
	f(x, y) = Halide::cast<float>(x * x + y * y);

	// Apply auto scheduling
	schedule::auto_schedule_2d(f);

	// Verify correctness
	Halide::Runtime::Buffer<float> out(128, 128);
	f.realize(out);

	for (int row = 0; row < 128; ++row) {
		for (int col = 0; col < 128; ++col) {
			float expected = static_cast<float>(col * col + row * row);
			EXPECT_NEAR(out(col, row), expected, 1e-5f);
		}
	}
}
