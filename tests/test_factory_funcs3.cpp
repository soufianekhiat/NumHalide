#include <gtest/gtest.h>
#include <Halide.h>
#include "../src/numhalide_all.h"
using namespace numhalide;

// --- indices ---
TEST(FactoryFuncs3, Indices_1D) {
    // indices({3}) returns [result[0]] where result[0](x) = x
    shape_t s = {3};
    auto idx = indices(s);
    ASSERT_EQ((int)idx.size(), 1);
    Halide::Runtime::Buffer<int32_t> out(3);
    idx[0].realize(out);
    EXPECT_EQ(out(0), 0);
    EXPECT_EQ(out(1), 1);
    EXPECT_EQ(out(2), 2);
}

TEST(FactoryFuncs3, Indices_2D_RowIndex) {
    // indices({3,4}): result[0](x,y) = y (row index)
    shape_t s = {3, 4};
    auto idx = indices(s);
    ASSERT_EQ((int)idx.size(), 2);
    // Halide Buffer(cols=4, rows=3) → out(x, y) where x=col, y=row
    Halide::Runtime::Buffer<int32_t> out(4, 3);
    idx[0].realize(out); // shape dim 0 = rows → Halide dim 1 = y
    // At (col=0, row=0): row index=0; at (col=2, row=1): row index=1
    EXPECT_EQ(out(0, 0), 0);
    EXPECT_EQ(out(0, 1), 1);
    EXPECT_EQ(out(0, 2), 2);
    EXPECT_EQ(out(3, 0), 0); // same row regardless of col
}

TEST(FactoryFuncs3, Indices_2D_ColIndex) {
    // indices({3,4}): result[1](x,y) = x (col index)
    shape_t s = {3, 4};
    auto idx = indices(s);
    Halide::Runtime::Buffer<int32_t> out(4, 3);
    idx[1].realize(out); // shape dim 1 = cols → Halide dim 0 = x
    // At (col=0, row=0): col=0; at (col=2, row=1): col=2
    EXPECT_EQ(out(0, 0), 0);
    EXPECT_EQ(out(1, 0), 1);
    EXPECT_EQ(out(2, 0), 2);
    EXPECT_EQ(out(3, 0), 3);
    EXPECT_EQ(out(2, 2), 2); // same col regardless of row
}

TEST(FactoryFuncs3, Indices_2D_SumEqualsArange) {
    // For a 1x4 grid, row index is always 0, col index = [0,1,2,3]
    shape_t s = {1, 4};
    auto idx = indices(s);
    Halide::Runtime::Buffer<int32_t> row_out(4, 1), col_out(4, 1);
    idx[0].realize(row_out);
    idx[1].realize(col_out);
    for (int c = 0; c < 4; ++c) {
        EXPECT_EQ(row_out(c, 0), 0);
        EXPECT_EQ(col_out(c, 0), c);
    }
}
