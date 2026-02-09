#include "numhalide_all.h"
#include <gtest/gtest.h>

using namespace numhalide;

TEST(Shape, Construction) {
	shape_t s1 = { 1, 2, 3 };
	EXPECT_EQ(s1.rank, 3);
	EXPECT_EQ(s1[0], 1);
	EXPECT_EQ(s1[1], 2);
	EXPECT_EQ(s1[2], 3);

	shape_t s2;
	EXPECT_EQ(s2.rank, 0);
}

TEST(Shape, Equality) {
	shape_t s1 = { 2, 4 };
	shape_t s2 = { 2, 4 };
	shape_t s3 = { 2, 5 };
	shape_t s4 = { 2, 4, 1 };

	EXPECT_EQ(s1, s2);
	EXPECT_NE(s1, s3);
	EXPECT_NE(s1, s4);
}

TEST(Shape, NormalizedAxis) {
	EXPECT_EQ(normalized_axis(0, 3), 0);
	EXPECT_EQ(normalized_axis(1, 3), 1);
	EXPECT_EQ(normalized_axis(2, 3), 2);
	EXPECT_EQ(normalized_axis(-1, 3), 2);
	EXPECT_EQ(normalized_axis(-2, 3), 1);
	EXPECT_EQ(normalized_axis(-3, 3), 0);
}

TEST(Shape, CheckSameExcept) {
	shape_t s1 = { 2, 3, 4 };
	shape_t s2 = { 2, 5, 4 };
	shape_t s3 = { 2, 3, 5 };

	EXPECT_TRUE(check_same_except(s1, s2, 1));
	EXPECT_FALSE(check_same_except(s1, s2, 0));
	EXPECT_FALSE(check_same_except(s1, s3, 1));
}

TEST(Shape, InferConcat) {
	shape_t s1 = { 2, 3 };
	shape_t s2 = { 2, 4 };
	
	shape_t res = infer_concat(s1, s2, 1);
	EXPECT_EQ(res.rank, 2);
	EXPECT_EQ(res[0], 2);
	EXPECT_EQ(res[1], 7);
}

TEST(Shape, InferBroadcast) {
	shape_t s1 = { 3, 1 };
	shape_t s2 = { 1, 4 };
	
	shape_t res = infer_broadcast(s1, s2);
	EXPECT_EQ(res.rank, 2);
	EXPECT_EQ(res[0], 3);
	EXPECT_EQ(res[1], 4);
	
	shape_t s3 = { 3 };
	shape_t s4 = { 2, 3 };
	res = infer_broadcast(s3, s4);
	EXPECT_EQ(res.rank, 2);
	EXPECT_EQ(res[0], 2);
	EXPECT_EQ(res[1], 3);
}
