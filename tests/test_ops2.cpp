#include <gtest/gtest.h>
#include <Halide.h>
#include "../src/numhalide_all.h"
using namespace numhalide;

static Halide::Func make_const_1d(std::initializer_list<float> vals, const std::string& n = "f") {
    std::vector<float> v(vals);
    Halide::Buffer<float> buf((int)v.size());
    for (int i = 0; i < (int)v.size(); ++i) buf(i) = v[i];
    Halide::Func f(n);
    Halide::Var x;
    f(x) = buf(x);
    return f;
}

static Halide::Func make_const_1d_int(std::initializer_list<int> vals, const std::string& n = "f") {
    std::vector<int> v(vals);
    Halide::Buffer<int32_t> buf((int)v.size());
    for (int i = 0; i < (int)v.size(); ++i) buf(i) = v[i];
    Halide::Func f(n);
    Halide::Var x;
    f(x) = buf(x);
    return f;
}

// --- choose ---
TEST(Ops2, Choose_Basic) {
    // indices=[0,1,0], choices=[[1,2,3],[4,5,6]]
    // result=[1,5,3]
    auto idx  = make_const_1d_int({0, 1, 0}, "idx");
    auto c0   = make_const_1d({1.0f, 2.0f, 3.0f}, "c0");
    auto c1   = make_const_1d({4.0f, 5.0f, 6.0f}, "c1");
    shape_t s = {3};
    auto r = choose(idx, {c0, c1}, s);
    Halide::Runtime::Buffer<float> out(3);
    r.realize(out);
    EXPECT_NEAR(out(0), 1.0f, 1e-5f);
    EXPECT_NEAR(out(1), 5.0f, 1e-5f);
    EXPECT_NEAR(out(2), 3.0f, 1e-5f);
}

TEST(Ops2, Choose_ThreeChoices) {
    auto idx = make_const_1d_int({0, 2, 1, 0}, "idx");
    auto c0  = make_const_1d({10.0f, 11.0f, 12.0f, 13.0f}, "c0");
    auto c1  = make_const_1d({20.0f, 21.0f, 22.0f, 23.0f}, "c1");
    auto c2  = make_const_1d({30.0f, 31.0f, 32.0f, 33.0f}, "c2");
    shape_t s = {4};
    auto r = choose(idx, {c0, c1, c2}, s);
    Halide::Runtime::Buffer<float> out(4);
    r.realize(out);
    EXPECT_NEAR(out(0), 10.0f, 1e-5f); // c0[0]
    EXPECT_NEAR(out(1), 31.0f, 1e-5f); // c2[1]
    EXPECT_NEAR(out(2), 22.0f, 1e-5f); // c1[2]
    EXPECT_NEAR(out(3), 13.0f, 1e-5f); // c0[3]
}

// --- piecewise ---
TEST(Ops2, Piecewise_Basic) {
    // x = [0, 1, 2, 3], conditions: x==0, x==1, x==2, x==3
    // values: 10, 20, 30, 40 → result = [10,20,30,40]
    Halide::Func x_f("x");
    Halide::Var xi;
    Halide::Buffer<int32_t> xbuf(4);
    for (int i = 0; i < 4; ++i) xbuf(i) = i;
    x_f(xi) = xbuf(xi);

    // conditions as int Funcs (1 where condition met)
    std::vector<Halide::Func> conds, vals;
    for (int k = 0; k < 4; ++k) {
        Halide::Func c("cond" + std::to_string(k));
        c(xi) = Halide::cast<int32_t>(x_f(xi) == k);
        conds.push_back(c);

        Halide::Func v("val" + std::to_string(k));
        v(xi) = Halide::cast<float>((k + 1) * 10);
        vals.push_back(v);
    }

    shape_t s = {4};
    auto r = piecewise(conds, vals, s);
    Halide::Runtime::Buffer<float> out(4);
    r.realize(out);
    EXPECT_NEAR(out(0), 10.0f, 1e-5f);
    EXPECT_NEAR(out(1), 20.0f, 1e-5f);
    EXPECT_NEAR(out(2), 30.0f, 1e-5f);
    EXPECT_NEAR(out(3), 40.0f, 1e-5f);
}

TEST(Ops2, Piecewise_StepFunction) {
    // x = [0, 1, 2, 3]: negative→0, non-negative→1
    Halide::Var xi;
    Halide::Buffer<float> xbuf(4);
    xbuf(0) = -1.0f; xbuf(1) = 0.0f; xbuf(2) = 1.5f; xbuf(3) = 2.0f;
    Halide::Func xf("xf");
    xf(xi) = xbuf(xi);

    Halide::Func neg_cond("neg_cond");
    neg_cond(xi) = Halide::cast<int32_t>(xf(xi) < 0.0f);
    Halide::Func pos_cond("pos_cond");
    pos_cond(xi) = Halide::cast<int32_t>(xf(xi) >= 0.0f);

    Halide::Func zero_val("zero"), one_val("one");
    zero_val(xi) = 0.0f;
    one_val(xi)  = 1.0f;

    shape_t s = {4};
    auto r = piecewise({neg_cond, pos_cond}, {zero_val, one_val}, s);
    Halide::Runtime::Buffer<float> out(4);
    r.realize(out);
    EXPECT_NEAR(out(0), 0.0f, 1e-5f);
    EXPECT_NEAR(out(1), 1.0f, 1e-5f);
    EXPECT_NEAR(out(2), 1.0f, 1e-5f);
    EXPECT_NEAR(out(3), 1.0f, 1e-5f);
}
