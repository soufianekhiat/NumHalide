/// @file test_random_ext.cpp
/// @brief Tests for extended random distributions

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

TEST(RandExt, ExponentialPositive) {
	shape_t s = { 64 };
	Halide::Func result = rand_exponential(Halide::Float(32), s, 1.0f, 42);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	for (int i = 0; i < 64; ++i) {
		EXPECT_GE(out(i), 0.0f);
	}
}

TEST(RandExt, BernoulliValues) {
	shape_t s = { 64 };
	Halide::Func result = rand_bernoulli(Halide::Float(32), s, 0.5f, 42);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	for (int i = 0; i < 64; ++i) {
		EXPECT_TRUE(out(i) == 0.0f || out(i) == 1.0f);
	}
}

TEST(RandExt, BernoulliMean) {
	shape_t s = { 1024 };
	Halide::Func result = rand_bernoulli(Halide::Float(32), s, 0.3f, 99);

	Halide::Runtime::Buffer<float> out(s.extents[0]);
	result.realize(out);

	float sum = 0;
	for (int i = 0; i < 1024; ++i) sum += out(i);
	float mean = sum / 1024.0f;

	// Should be roughly 0.3
	EXPECT_NEAR(mean, 0.3f, 0.1f);
}

TEST(RandExt, ChoiceRange) {
	shape_t s = { 64 };
	int n = 10;
	Halide::Func result = rand_choice(Halide::Int(32), s, n, 42);

	Halide::Runtime::Buffer<int32_t> out(s.extents[0]);
	result.realize(out);

	for (int i = 0; i < 64; ++i) {
		EXPECT_GE(out(i), 0);
		EXPECT_LT(out(i), n);
	}
}

TEST(RandExt, Reproducible) {
	shape_t s = { 16 };
	Halide::Func r1 = rand_exponential(Halide::Float(32), s, 2.0f, 123);
	Halide::Func r2 = rand_exponential(Halide::Float(32), s, 2.0f, 123);

	Halide::Runtime::Buffer<float> out1(s.extents[0]);
	Halide::Runtime::Buffer<float> out2(s.extents[0]);
	r1.realize(out1);
	r2.realize(out2);

	for (int i = 0; i < 16; ++i) {
		EXPECT_NEAR(out1(i), out2(i), 1e-5f);
	}
}
