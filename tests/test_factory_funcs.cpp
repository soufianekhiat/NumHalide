/// @file test_factory_funcs.cpp
/// @brief Basic tests for factory functions (Milestone 0)

#include <gtest/gtest.h>
#include "../src/numhalide_all.h"

using namespace numhalide;

/// Test linspace with basic parameters
TEST(FactoryFuncs, Linspace) {
	// Create linspace from 0 to 1 with 5 points
	Halide::Func f = linspace(Halide::Float(32), 0.0f, 1.0f, 5);
	Halide::Runtime::Buffer<float> result(5);
	f.realize(result);

	// Verify values
	EXPECT_NEAR(result(0), 0.0f, 1e-5f);
	EXPECT_NEAR(result(1), 0.25f, 1e-5f);
	EXPECT_NEAR(result(2), 0.5f, 1e-5f);
	EXPECT_NEAR(result(3), 0.75f, 1e-5f);
	EXPECT_NEAR(result(4), 1.0f, 1e-5f);
}

/// Test linspace without endpoint
TEST(FactoryFuncs, LinspaceNoEndpoint) {
	// Create linspace from 0 to 1 with 5 points, endpoint=false
	Halide::Func f = linspace(Halide::Float(32), 0.0f, 1.0f, 5, false);
	Halide::Runtime::Buffer<float> result(5);
	f.realize(result);

	// Verify values (step = 1.0/5 = 0.2)
	EXPECT_NEAR(result(0), 0.0f, 1e-5f);
	EXPECT_NEAR(result(1), 0.2f, 1e-5f);
	EXPECT_NEAR(result(2), 0.4f, 1e-5f);
	EXPECT_NEAR(result(3), 0.6f, 1e-5f);
	EXPECT_NEAR(result(4), 0.8f, 1e-5f);
}

/// Test zeros function
TEST(FactoryFuncs, Zeros) {
	// Create 2D array of zeros
	Halide::Func f = zeros(Halide::Float(32), 2);
	Halide::Runtime::Buffer<float> result(3, 3);
	f.realize(result);

	// Verify all values are zero
	for (int y = 0; y < 3; ++y) {
		for (int x = 0; x < 3; ++x) {
			EXPECT_NEAR(result(x, y), 0.0f, 1e-5f);
		}
	}
}

/// Test ones function
TEST(FactoryFuncs, Ones) {
	// Create 2D array of ones
	Halide::Func f = ones(Halide::Float(32), 2);
	Halide::Runtime::Buffer<float> result(4, 4);
	f.realize(result);

	// Verify all values are one
	for (int y = 0; y < 4; ++y) {
		for (int x = 0; x < 4; ++x) {
			EXPECT_NEAR(result(x, y), 1.0f, 1e-5f);
		}
	}
}

/// Test full function
TEST(FactoryFuncs, Full) {
	// Create 2D array filled with 3.14
	Halide::Func f = full(Halide::Float(32), 3.14f, 2);
	Halide::Runtime::Buffer<float> result(3, 3);
	f.realize(result);

	// Verify all values are 3.14
	for (int y = 0; y < 3; ++y) {
		for (int x = 0; x < 3; ++x) {
			EXPECT_NEAR(result(x, y), 3.14f, 1e-5f);
		}
	}
}

/// Test arange function
TEST(FactoryFuncs, Arange) {
	// Create arange from 0 to 5
	Halide::Func f = arange(Halide::Float(32), 5.0f);
	Halide::Runtime::Buffer<float> result(5);
	f.realize(result);

	// Verify values
	for (int i = 0; i < 5; ++i) {
		EXPECT_NEAR(result(i), (float)i, 1e-5f);
	}
}

/// Test arange with step
TEST(FactoryFuncs, ArangeStep) {
	// Create arange from 0 to 5 with step 0.5
	Halide::Func f = arange(Halide::Float(32), 0.0f, 5.0f, 0.5f);
	Halide::Runtime::Buffer<float> result(10);
	f.realize(result);

	// Verify values
	for (int i = 0; i < 10; ++i) {
		EXPECT_NEAR(result(i), i * 0.5f, 1e-5f);
	}
}

/// Test zeros with shape_t
TEST(FactoryFuncs, ZerosShape) {
	shape_t s = { 3, 4 };
	Halide::Func f = zeros(Halide::Float(32), s);
	Halide::Runtime::Buffer<float> result(3, 4);
	f.realize(result);

	for (int y = 0; y < 4; ++y) {
		for (int x = 0; x < 3; ++x) {
			EXPECT_NEAR(result(x, y), 0.0f, 1e-5f);
		}
	}
}

/// Test ones with shape_t
TEST(FactoryFuncs, OnesShape) {
	shape_t s = { 2, 2 };
	Halide::Func f = ones(Halide::Float(32), s);
	Halide::Runtime::Buffer<float> result(2, 2);
	f.realize(result);

	for (int y = 0; y < 2; ++y) {
		for (int x = 0; x < 2; ++x) {
			EXPECT_NEAR(result(x, y), 1.0f, 1e-5f);
		}
	}
}

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
