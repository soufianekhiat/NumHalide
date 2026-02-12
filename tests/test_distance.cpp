/// @file test_distance.cpp
/// @brief Tests for distance computations

#include <gtest/gtest.h>
#include "numhalide_all.h"
#include <cmath>

using namespace numhalide;

TEST(Distance, EuclideanKnown) {
	// Two points: a=[0,0], b=[3,4] -> distance = 5
	Halide::Func a("a"), b("b");
	Halide::Var d, i;
	a(d, i) = 0.0f;  // single point at origin
	b(d, i) = Halide::select(d == 0, 3.0f, 4.0f);  // point (3,4)

	Halide::Func result = cdist_euclidean(a, b, 1, 1, 2);

	Halide::Runtime::Buffer<float> out(1, 1);
	result.realize(out);

	EXPECT_NEAR(out(0, 0), 5.0f, 1e-4f);
}

TEST(Distance, ManhattanKnown) {
	Halide::Func a("a"), b("b");
	Halide::Var d, i;
	a(d, i) = 0.0f;
	b(d, i) = Halide::select(d == 0, 3.0f, 4.0f);

	Halide::Func result = cdist_manhattan(a, b, 1, 1, 2);

	Halide::Runtime::Buffer<float> out(1, 1);
	result.realize(out);

	EXPECT_NEAR(out(0, 0), 7.0f, 1e-4f);  // |3| + |4|
}

TEST(Distance, EuclideanSelf) {
	// Distance to self should be 0
	Halide::Func a("a");
	Halide::Var d, i;
	a(d, i) = Halide::cast<float>(d + i);  // 2 points in 2D

	Halide::Func result = cdist_euclidean(a, a, 2, 2, 2);

	Halide::Runtime::Buffer<float> out(2, 2);
	result.realize(out);

	EXPECT_NEAR(out(0, 0), 0.0f, 1e-4f);  // d(a0, a0)
	EXPECT_NEAR(out(1, 1), 0.0f, 1e-4f);  // d(a1, a1)
}

TEST(Distance, EuclideanSymmetric) {
	Halide::Func a("a"), b("b");
	Halide::Var d, i;
	a(d, i) = Halide::cast<float>(d);
	b(d, i) = Halide::cast<float>(d + 1);

	Halide::Func d_ab = cdist_euclidean(a, b, 1, 1, 2);
	Halide::Func d_ba = cdist_euclidean(b, a, 1, 1, 2);

	Halide::Runtime::Buffer<float> out_ab(1, 1), out_ba(1, 1);
	d_ab.realize(out_ab);
	d_ba.realize(out_ba);

	EXPECT_NEAR(out_ab(0, 0), out_ba(0, 0), 1e-4f);
}

TEST(Distance, CosineSameDir) {
	shape_t s = { 2 };
	Halide::Func a("a"), b("b");
	Halide::Var x;
	a(x) = Halide::select(x == 0, 1.0f, 0.0f);
	b(x) = Halide::select(x == 0, 2.0f, 0.0f);  // same direction

	Halide::Func result = cosine_similarity(a, b, s);

	Halide::Runtime::Buffer<float> out(1);
	result.realize(out);

	EXPECT_NEAR(out(0), 1.0f, 1e-4f);
}

TEST(Distance, CosineOrthogonal) {
	shape_t s = { 2 };
	Halide::Func a("a"), b("b");
	Halide::Var x;
	a(x) = Halide::select(x == 0, 1.0f, 0.0f);
	b(x) = Halide::select(x == 0, 0.0f, 1.0f);  // orthogonal

	Halide::Func result = cosine_similarity(a, b, s);

	Halide::Runtime::Buffer<float> out(1);
	result.realize(out);

	EXPECT_NEAR(out(0), 0.0f, 1e-4f);
}
