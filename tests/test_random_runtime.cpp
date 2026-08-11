/// @file test_random_runtime.cpp
/// @brief Tests for the RUNTIME typed 1-D random overloads (Var-indexed
/// random_float stream, no shape/seed params): rand_uniform, rand_bernoulli,
/// rand_exponential, rand_choice. The stream is Halide's stateless RNG, so
/// exact draws are pipeline-construction-order dependent — all tests are
/// STATISTICAL (bounds + moments on 4096-sample realizations), never
/// exact-value pins, plus one typing pin for the f64 path.

#include <gtest/gtest.h>
#include "numhalide_all.h"

#include <cmath>

using namespace numhalide;

namespace {

constexpr int kN = 4096;

} // namespace

TEST(RandomRuntime, UniformBoundsAndMean) {
	// U[2, 5): every sample inside the half-open interval, mean near the
	// midpoint 3.5. sd = 3/sqrt(12) ~ 0.866, sem ~ 0.0135 — the 0.15
	// tolerance is ~11 sigma.
	Halide::Func rt = rand_uniform(Halide::Float(32), 2.0f, 5.0f, "rtu");
	Halide::Runtime::Buffer<float> out(kN);
	rt.realize(out);

	double sum = 0.0;
	for (int i = 0; i < kN; ++i) {
		ASSERT_GE(out(i), 2.0f) << "sample " << i << " below low";
		ASSERT_LT(out(i), 5.0f) << "sample " << i << " at/above high";
		sum += out(i);
	}
	EXPECT_NEAR(sum / kN, 3.5, 0.15);
}

TEST(RandomRuntime, UniformF64TypePin) {
	// The affine map is computed in the requested type: Float(64) in,
	// Float(64) out. Realize a small f64 buffer to prove the pipeline
	// actually runs at that type, and bounds-check it.
	Halide::Func rt = rand_uniform(Halide::Float(64),
	                               Halide::Expr(0.0), Halide::Expr(1.0),
	                               "rtu64");
	ASSERT_EQ(rt.types().size(), (size_t)1);
	EXPECT_EQ(rt.types()[0], Halide::Float(64));

	Halide::Runtime::Buffer<double> out(64);
	rt.realize(out);
	for (int i = 0; i < 64; ++i) {
		ASSERT_GE(out(i), 0.0);
		ASSERT_LT(out(i), 1.0);
	}
}

TEST(RandomRuntime, BernoulliBinaryAndMean) {
	// Every sample is EXACTLY 0.0f or 1.0f; mean near p = 0.3.
	// sem = sqrt(0.3 * 0.7) / 64 ~ 0.0072 — the 0.05 tolerance is ~7 sigma.
	Halide::Func rt = rand_bernoulli(Halide::Float(32), 0.3f, "rtb");
	Halide::Runtime::Buffer<float> out(kN);
	rt.realize(out);

	double sum = 0.0;
	for (int i = 0; i < kN; ++i) {
		ASSERT_TRUE(out(i) == 0.0f || out(i) == 1.0f)
			<< "sample " << i << " = " << out(i) << " is not binary";
		sum += out(i);
	}
	EXPECT_NEAR(sum / kN, 0.3, 0.05);
}

TEST(RandomRuntime, ExponentialNonnegAndMean) {
	// Exp(lambda = 2): all samples strictly positive and, because of the
	// documented log(0)-guard clamp(u, 0.001, 0.999), bounded above by
	// -ln(0.001)/lambda ~ 3.454. Mean near 1/lambda = 0.5; the clamp biases
	// the true mean ~0.1% low (~0.4995), far inside the statistical
	// tolerance (sd = 0.5, sem ~ 0.0078 — 0.08 is ~10 sigma).
	float const lambda = 2.0f;
	Halide::Func rt = rand_exponential(Halide::Float(32), lambda, "rte");
	Halide::Runtime::Buffer<float> out(kN);
	rt.realize(out);

	float const upper = -std::log(0.001f) / lambda;
	double sum = 0.0;
	for (int i = 0; i < kN; ++i) {
		ASSERT_GT(out(i), 0.0f) << "sample " << i << " not positive";
		ASSERT_LE(out(i), upper + 1e-4f)
			<< "sample " << i << " above the clamp-truncated tail";
		sum += out(i);
	}
	EXPECT_NEAR(sum / kN, 0.5, 0.08);
}

TEST(RandomRuntime, ChoiceRangeAndAllBinsHit) {
	// [0, 6): every sample in range, Int32 output, and all 6 bins hit —
	// P(any bin empty) = 6 * (5/6)^4096 ~ 0.
	Halide::Func rt = rand_choice(6, "rtc");
	ASSERT_EQ(rt.types().size(), (size_t)1);
	EXPECT_EQ(rt.types()[0], Halide::Int(32));

	Halide::Runtime::Buffer<int> out(kN);
	rt.realize(out);

	int histogram[6] = {0, 0, 0, 0, 0, 0};
	for (int i = 0; i < kN; ++i) {
		ASSERT_GE(out(i), 0) << "sample " << i << " negative";
		ASSERT_LT(out(i), 6) << "sample " << i << " at/above n";
		histogram[out(i)] += 1;
	}
	for (int b = 0; b < 6; ++b)
		EXPECT_GT(histogram[b], 0) << "bin " << b << " never hit";
}
