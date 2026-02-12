/// @file test_random.cpp
/// @brief Tests for random number generation and *_like functions

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

// -----------------------------------------------------------------------------
// rand_uniform Tests
// -----------------------------------------------------------------------------

TEST(Random, UniformRange) {
	// Check that all values are in [0, 1)
	shape_t s = { 100, 100 };
	Halide::Func r = rand_uniform(Halide::Float(32), s, 42);

	Halide::Runtime::Buffer<float> out(s.extents[1], s.extents[0]);
	r.realize(out);

	for (int y = 0; y < s.extents[0]; ++y) {
		for (int x = 0; x < s.extents[1]; ++x) {
			float val = out(x, y);
			EXPECT_GE(val, 0.0f) << "Value at (" << x << "," << y << ") is below 0";
			EXPECT_LT(val, 1.0f) << "Value at (" << x << "," << y << ") is >= 1";
		}
	}
}

TEST(Random, UniformMean) {
	// Statistical test: mean should be approximately 0.5
	shape_t s = { 1000 };
	Halide::Func r = rand_uniform(Halide::Float(32), s, 123);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	r.realize(out);

	double sum = 0.0;
	for (int i = 0; i < s.extents[0]; ++i) {
		sum += out(i);
	}
	double mean = sum / s.extents[0];

	// Mean should be close to 0.5 (allow some tolerance)
	EXPECT_NEAR(mean, 0.5, 0.05);
}

TEST(Random, UniformReproducibility) {
	// Same Func should produce same values when realized multiple times
	shape_t s = { 10, 10 };
	Halide::Func r = rand_uniform(Halide::Float(32), s, 999);

	Halide::Runtime::Buffer<float> out1(s.extents[1], s.extents[0]);
	Halide::Runtime::Buffer<float> out2(s.extents[1], s.extents[0]);
	r.realize(out1);
	r.realize(out2);

	for (int y = 0; y < s.extents[0]; ++y) {
		for (int x = 0; x < s.extents[1]; ++x) {
			EXPECT_EQ(out1(x, y), out2(x, y));
		}
	}
}

TEST(Random, UniformDifferentSeeds) {
	// Different seeds should produce different values
	shape_t s = { 10, 10 };
	Halide::Func r1 = rand_uniform(Halide::Float(32), s, 100);
	Halide::Func r2 = rand_uniform(Halide::Float(32), s, 200);

	Halide::Runtime::Buffer<float> out1(s.extents[1], s.extents[0]);
	Halide::Runtime::Buffer<float> out2(s.extents[1], s.extents[0]);
	r1.realize(out1);
	r2.realize(out2);

	int different_count = 0;
	for (int y = 0; y < s.extents[0]; ++y) {
		for (int x = 0; x < s.extents[1]; ++x) {
			if (out1(x, y) != out2(x, y)) {
				different_count++;
			}
		}
	}
	// Most values should be different
	EXPECT_GT(different_count, 50);
}

TEST(Random, UniformCustomRange) {
	// Test uniform in [low, high) range
	shape_t s = { 100 };
	float low = 5.0f;
	float high = 10.0f;
	Halide::Func r = rand_uniform(Halide::Float(32), s, low, high, 42);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	r.realize(out);

	for (int i = 0; i < s.extents[0]; ++i) {
		float val = out(i);
		EXPECT_GE(val, low);
		EXPECT_LT(val, high);
	}
}

// -----------------------------------------------------------------------------
// rand_normal Tests
// -----------------------------------------------------------------------------

TEST(Random, NormalMeanStddev) {
	// Statistical test: mean ~0, stddev ~1 for standard normal
	shape_t s = { 10000 };
	Halide::Func r = rand_normal(Halide::Float(32), s, 0.0f, 1.0f, 42);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	r.realize(out);

	// Compute mean
	double sum = 0.0;
	for (int i = 0; i < s.extents[0]; ++i) {
		sum += out(i);
	}
	double mean = sum / s.extents[0];

	// Compute variance
	double var_sum = 0.0;
	for (int i = 0; i < s.extents[0]; ++i) {
		double diff = out(i) - mean;
		var_sum += diff * diff;
	}
	double stddev = std::sqrt(var_sum / s.extents[0]);

	// Mean should be close to 0, stddev close to 1
	EXPECT_NEAR(mean, 0.0, 0.1);
	EXPECT_NEAR(stddev, 1.0, 0.1);
}

TEST(Random, NormalCustomMeanStddev) {
	// Test with custom mean and stddev
	shape_t s = { 10000 };
	float target_mean = 5.0f;
	float target_stddev = 2.0f;
	Halide::Func r = rand_normal(Halide::Float(32), s, target_mean, target_stddev, 42);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	r.realize(out);

	// Compute mean
	double sum = 0.0;
	for (int i = 0; i < s.extents[0]; ++i) {
		sum += out(i);
	}
	double mean = sum / s.extents[0];

	// Compute variance
	double var_sum = 0.0;
	for (int i = 0; i < s.extents[0]; ++i) {
		double diff = out(i) - mean;
		var_sum += diff * diff;
	}
	double stddev = std::sqrt(var_sum / s.extents[0]);

	EXPECT_NEAR(mean, target_mean, 0.2);
	EXPECT_NEAR(stddev, target_stddev, 0.2);
}

// -----------------------------------------------------------------------------
// rand_int Tests
// -----------------------------------------------------------------------------

TEST(Random, IntRange) {
	// Check that all values are in [low, high)
	shape_t s = { 1000 };
	int low = 0;
	int high = 10;
	Halide::Func r = rand_int(Halide::Int(32), s, low, high, 42);

	Halide::Runtime::Buffer<int32_t> out(s.extents[0]);
	r.realize(out);

	for (int i = 0; i < s.extents[0]; ++i) {
		int val = out(i);
		EXPECT_GE(val, low);
		EXPECT_LT(val, high);
	}
}

TEST(Random, IntDistribution) {
	// Check that all values in range appear (roughly uniform)
	shape_t s = { 10000 };
	int low = 0;
	int high = 10;
	Halide::Func r = rand_int(Halide::Int(32), s, low, high, 42);

	Halide::Runtime::Buffer<int32_t> out(s.extents[0]);
	r.realize(out);

	// Count occurrences
	int counts[10] = {0};
	for (int i = 0; i < s.extents[0]; ++i) {
		int val = out(i);
		if (val >= 0 && val < 10) {
			counts[val]++;
		}
	}

	// Each value should appear at least some times (roughly 1000 each)
	for (int i = 0; i < 10; ++i) {
		EXPECT_GT(counts[i], 500) << "Value " << i << " appears too rarely";
	}
}

// -----------------------------------------------------------------------------
// *_like Tests
// -----------------------------------------------------------------------------

TEST(Like, ZerosLike) {
	shape_t s = { 3, 4 };
	Halide::Func original("original");
	Halide::Var x, y;
	original(x, y) = Halide::cast<float>(x + y);

	Halide::Func result = zeros_like(original, s);

	Halide::Runtime::Buffer<float> out(s.extents[1], s.extents[0]);
	result.realize(out);

	for (int y = 0; y < s.extents[0]; ++y) {
		for (int x = 0; x < s.extents[1]; ++x) {
			EXPECT_EQ(out(x, y), 0.0f);
		}
	}
}

TEST(Like, OnesLike) {
	shape_t s = { 3, 4 };
	Halide::Func original("original");
	Halide::Var x, y;
	original(x, y) = Halide::cast<float>(x + y);

	Halide::Func result = ones_like(original, s);

	Halide::Runtime::Buffer<float> out(s.extents[1], s.extents[0]);
	result.realize(out);

	for (int y = 0; y < s.extents[0]; ++y) {
		for (int x = 0; x < s.extents[1]; ++x) {
			EXPECT_EQ(out(x, y), 1.0f);
		}
	}
}

TEST(Like, FullLike) {
	shape_t s = { 3, 4 };
	Halide::Func original("original");
	Halide::Var x, y;
	original(x, y) = Halide::cast<float>(x + y);

	float fill_value = 42.0f;
	Halide::Func result = full_like(original, s, fill_value);

	Halide::Runtime::Buffer<float> out(s.extents[1], s.extents[0]);
	result.realize(out);

	for (int y = 0; y < s.extents[0]; ++y) {
		for (int x = 0; x < s.extents[1]; ++x) {
			EXPECT_EQ(out(x, y), fill_value);
		}
	}
}

TEST(Like, PreservesType) {
	// Test that *_like preserves the type of the original
	shape_t s = { 2, 2 };
	Halide::Func original("original");
	Halide::Var x, y;
	original(x, y) = Halide::cast<int32_t>(x + y);

	Halide::Func result = zeros_like(original, s);

	// The result should have int32 type
	EXPECT_EQ(result.types()[0], Halide::Int(32));
}
