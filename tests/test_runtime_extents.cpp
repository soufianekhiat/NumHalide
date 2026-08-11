/// @file test_runtime_extents.cpp
/// @brief Every runtime-Expr overload must agree with its compile-time
///        (shape_t / int) form on identical data
///
/// Two flavors are covered, called out per test:
///  - TRUE overloads, where a separate Expr-taking function was added next to
///    the shape_t/int one: trace, norm, frobenius_norm, matmul, matvec, kron,
///    polymul. Passing a literal int would still pick the int/shape_t form,
///    so these tests pass Halide::Expr(...) explicitly.
///  - Parameters that simply BECAME Expr (no second overload): sort_1d,
///    argsort_1d, inner_1d, and the window functions. Plain int arguments
///    convert implicitly and land in the same body; these are value-pinned.

#include <gtest/gtest.h>
#include "numhalide_all.h"

#include <cmath>

using namespace numhalide;

namespace {

Halide::Func make_matf(int rows, int cols,
	std::initializer_list<float> vals, const std::string& n)
{
	std::vector<float> v(vals);
	Halide::Buffer<float> buf(cols, rows);
	for (int r = 0; r < rows; ++r)
		for (int c = 0; c < cols; ++c)
			buf(c, r) = v[(size_t)(r * cols + c)];
	Halide::Func f(n);
	Halide::Var x, y;
	f(x, y) = buf(x, y);
	return f;
}

Halide::Func make_vecf(std::initializer_list<float> vals, const std::string& n)
{
	std::vector<float> v(vals);
	Halide::Buffer<float> buf((int)v.size());
	for (int i = 0; i < (int)v.size(); ++i)
		buf(i) = v[i];
	Halide::Func f(n);
	Halide::Var x;
	f(x) = buf(x);
	return f;
}

} // namespace

TEST(RuntimeExtents, TraceExprVsShape) {
	// True overload: trace(Func, Expr) vs trace(Func, shape_t)
	auto mat = make_matf(3, 3, { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, "m_tr");

	auto ct = trace(mat, shape_t{ 3, 3 }, "tr_ct");
	auto rt = trace(mat, Halide::Expr(3), "tr_rt");

	Halide::Runtime::Buffer<float> out_ct(1), out_rt(1);
	ct.realize(out_ct);
	rt.realize(out_rt);

	EXPECT_NEAR(out_ct(0), 15.0f, 1e-6f);
	EXPECT_NEAR(out_rt(0), out_ct(0), 1e-6f);
}

TEST(RuntimeExtents, NormExprVsShape) {
	// True overload: norm(Func, Expr) vs norm(Func, shape_t)
	// [1,2,3,4] -> sqrt(30)
	auto vec = make_vecf({ 1.0f, 2.0f, 3.0f, 4.0f }, "v_nrm");

	auto ct = norm(vec, shape_t{ 4 }, "nrm_ct");
	auto rt = norm(vec, Halide::Expr(4), "nrm_rt");

	Halide::Runtime::Buffer<float> out_ct(1), out_rt(1);
	ct.realize(out_ct);
	rt.realize(out_rt);

	EXPECT_NEAR(out_ct(0), std::sqrt(30.0f), 1e-5f);
	EXPECT_NEAR(out_rt(0), out_ct(0), 1e-6f);
}

TEST(RuntimeExtents, FrobeniusNormExprVsShape) {
	// True overload. Fixture is 2 rows x 3 cols (shape {2,3}), sum of squares
	// 1+4+9+16+25+36 = 91 -> sqrt(91).
	// DERIVED FROM SOURCE (la.h): the runtime overload's (m, n) are
	// BUFFER-AXIS extents — RDom r(0, m, 0, n) reads mat(r.x, r.y), so m is
	// the x (col) extent and n the y (row) extent. For this {rows=2, cols=3}
	// matrix the matching call is therefore (Expr(3), Expr(2)); the shape
	// form with {2,3} builds RDom(0, N=3, 0, M=2) — the identical region.
	auto mat = make_matf(2, 3, { 1, 2, 3, 4, 5, 6 }, "m_fro");

	auto ct = frobenius_norm(mat, shape_t{ 2, 3 }, "fro_ct");
	auto rt = frobenius_norm(mat, Halide::Expr(3), Halide::Expr(2), "fro_rt");

	Halide::Runtime::Buffer<float> out_ct(1), out_rt(1);
	ct.realize(out_ct);
	rt.realize(out_rt);

	EXPECT_NEAR(out_ct(0), std::sqrt(91.0f), 1e-4f);
	EXPECT_NEAR(out_rt(0), out_ct(0), 1e-6f);
}

TEST(RuntimeExtents, Inner1D) {
	// Parameter is Expr (no int overload): a plain int converts implicitly.
	// [1,2,3] . [4,5,6] = 32
	auto a = make_vecf({ 1.0f, 2.0f, 3.0f }, "a_in");
	auto b = make_vecf({ 4.0f, 5.0f, 6.0f }, "b_in");

	auto r = inner_1d(a, b, 3, "inner_rt");
	Halide::Runtime::Buffer<float> out(1);
	r.realize(out);
	EXPECT_NEAR(out(0), 32.0f, 1e-5f);
}

TEST(RuntimeExtents, Sort1DAndArgsort1DExprN) {
	// n is Expr (no int overload); [0.3, 0.1, 0.2]
	auto f = make_vecf({ 0.3f, 0.1f, 0.2f }, "f_srt");

	auto s = sort_1d(f, Halide::Expr(3), true, "sort_rt");
	Halide::Runtime::Buffer<float> out_s(3);
	s.realize(out_s);
	EXPECT_NEAR(out_s(0), 0.1f, 1e-6f);
	EXPECT_NEAR(out_s(1), 0.2f, 1e-6f);
	EXPECT_NEAR(out_s(2), 0.3f, 1e-6f);

	auto a = argsort_1d(f, Halide::Expr(3), true, "asort_rt");
	Halide::Runtime::Buffer<int32_t> out_a(3);
	a.realize(out_a);
	EXPECT_EQ(out_a(0), 1);
	EXPECT_EQ(out_a(1), 2);
	EXPECT_EQ(out_a(2), 0);
}

TEST(RuntimeExtents, MatmulExprKVsShape) {
	// True overload: matmul(a, b, Expr K) — library convention
	// ret(x, y) += a(k, y) * b(x, k) — vs matmul(a, sa, b, sb).
	// Same data as LA.Matmul2x3_3x2: expected [[58,64],[139,154]].
	shape_t sa = { 2, 3 };
	shape_t sb = { 3, 2 };

	Halide::Func a("a_mmrt");
	Halide::Func b("b_mmrt");
	Halide::Var x, y;
	a(x, y) = Halide::cast<float>(y * 3 + x + 1);
	b(x, y) = Halide::cast<float>(y * 2 + x + 7);

	auto ct = matmul(a, sa, b, sb, "mm_ct");
	auto rt = matmul(a, b, Halide::Expr(3), "mm_rt");

	Halide::Runtime::Buffer<float> out_ct(2, 2), out_rt(2, 2);
	ct.realize(out_ct);
	rt.realize(out_rt);

	EXPECT_NEAR(out_ct(0, 0),  58.0f, 1e-5f);
	EXPECT_NEAR(out_ct(1, 0),  64.0f, 1e-5f);
	EXPECT_NEAR(out_ct(0, 1), 139.0f, 1e-5f);
	EXPECT_NEAR(out_ct(1, 1), 154.0f, 1e-5f);
	for (int r = 0; r < 2; ++r)
		for (int c = 0; c < 2; ++c)
			EXPECT_NEAR(out_rt(c, r), out_ct(c, r), 1e-6f);
}

TEST(RuntimeExtents, MatvecExprNVsShape) {
	// True overload: matvec(mat, vec, Expr n) vs matvec(mat, smat, vec, svec)
	// [[1,2],[3,4],[5,6]] @ [7,8] = [23, 53, 83]
	shape_t smat = { 3, 2 };
	shape_t svec = { 2 };

	Halide::Func mat("m_mvrt");
	Halide::Func vec("v_mvrt");
	Halide::Var x, y;
	mat(x, y) = Halide::cast<float>(y * 2 + x + 1);
	vec(x) = Halide::cast<float>(x + 7);

	auto ct = matvec(mat, smat, vec, svec, "mv_ct");
	auto rt = matvec(mat, vec, Halide::Expr(2), "mv_rt");

	Halide::Runtime::Buffer<float> out_ct(3), out_rt(3);
	ct.realize(out_ct);
	rt.realize(out_rt);

	EXPECT_NEAR(out_ct(0), 23.0f, 1e-5f);
	EXPECT_NEAR(out_ct(1), 53.0f, 1e-5f);
	EXPECT_NEAR(out_ct(2), 83.0f, 1e-5f);
	for (int i = 0; i < 3; ++i)
		EXPECT_NEAR(out_rt(i), out_ct(i), 1e-6f);
}

TEST(RuntimeExtents, KronExprVsShape) {
	// True overload: kron(a, b, Expr b_d0, Expr b_d1) vs kron(a, sa, b, sb).
	// a: 2x2, b: 3 rows x 2 cols. b_d0/b_d1 are b's BUFFER-axis extents
	// (dim 0 = x = cols = 2, dim 1 = y = rows = 3), matching the compile-time
	// body a(col/Nb, row/Mb)*b(col%Nb, row%Mb) with Nb=2, Mb=3.
	shape_t s_a = { 2, 2 };
	shape_t s_b = { 3, 2 };
	auto a = make_matf(2, 2, { 1, 2, 3, 4 }, "a_kr");
	auto b = make_matf(3, 2, { 5, 6, 7, 8, 9, 10 }, "b_kr");

	auto ct = kron(a, s_a, b, s_b, "kr_ct");
	auto rt = kron(a, b, Halide::Expr(2), Halide::Expr(3), "kr_rt");

	shape_t sc = infer_kron(s_a, s_b);
	EXPECT_EQ(sc.extents[0], 6);
	EXPECT_EQ(sc.extents[1], 4);

	Halide::Runtime::Buffer<float> out_ct(sc.extents[1], sc.extents[0]);
	Halide::Runtime::Buffer<float> out_rt(sc.extents[1], sc.extents[0]);
	ct.realize(out_ct);
	rt.realize(out_rt);

	// kron[i*Mb + m, j*Nb + n] = A[i,j] * B[m,n]
	const double A[2][2] = { { 1, 2 }, { 3, 4 } };
	const double B[3][2] = { { 5, 6 }, { 7, 8 }, { 9, 10 } };
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j)
			for (int m = 0; m < 3; ++m)
				for (int n = 0; n < 2; ++n) {
					int row = i * 3 + m;
					int col = j * 2 + n;
					float expected = static_cast<float>(A[i][j] * B[m][n]);
					EXPECT_NEAR(out_ct(col, row), expected, 1e-5f)
						<< "kron ct at row=" << row << " col=" << col;
					EXPECT_NEAR(out_rt(col, row), out_ct(col, row), 1e-6f)
						<< "kron rt at row=" << row << " col=" << col;
				}
}

TEST(RuntimeExtents, WindowsExprVsInt) {
	// size became Expr (no int overload): int 5 converts implicitly and runs
	// the identical body. Compare both spellings, then pin blackman(5)
	// against the numpy symmetric window [~0, 0.34, 1.0, 0.34, ~0].
	struct Case {
		const char* name;
		Halide::Func rt;
		Halide::Func ct;
	};
	Case cases[] = {
		{ "hanning",  hanning(Halide::Expr(5), "han_rt"),  hanning(5, "han_ct") },
		{ "hamming",  hamming(Halide::Expr(5), "ham_rt"),  hamming(5, "ham_ct") },
		{ "blackman", blackman(Halide::Expr(5), "bl_rt"),  blackman(5, "bl_ct") },
		{ "bartlett", bartlett(Halide::Expr(5), "bar_rt"), bartlett(5, "bar_ct") },
	};

	for (auto& c : cases) {
		Halide::Runtime::Buffer<float> out_rt(5), out_ct(5);
		c.rt.realize(out_rt);
		c.ct.realize(out_ct);
		for (int i = 0; i < 5; ++i)
			EXPECT_NEAR(out_rt(i), out_ct(i), 1e-6f) << c.name << " at " << i;
	}

	Halide::Runtime::Buffer<float> bl(5);
	blackman(5, "bl_pin").realize(bl);
	EXPECT_NEAR(bl(0), 0.0f,  1e-3f);
	EXPECT_NEAR(bl(1), 0.34f, 1e-3f);
	EXPECT_NEAR(bl(2), 1.0f,  1e-3f);
	EXPECT_NEAR(bl(3), 0.34f, 1e-3f);
	EXPECT_NEAR(bl(4), 0.0f,  1e-3f);
}

TEST(RuntimeExtents, PolymulExprVsInt) {
	// True overload: polymul(a, Expr, b, Expr) vs polymul(a, int, b, int).
	// Highest-first: (x+1)(x-1) = x^2 - 1 -> [1, 0, -1]
	auto a = make_vecf({ 1.0f,  1.0f }, "a_pm");
	auto b = make_vecf({ 1.0f, -1.0f }, "b_pm");

	auto ct = polymul(a, 2, b, 2, "pm_ct");
	auto rt = polymul(a, Halide::Expr(2), b, Halide::Expr(2), "pm_rt");

	Halide::Runtime::Buffer<float> out_ct(3), out_rt(3);
	ct.realize(out_ct);
	rt.realize(out_rt);

	EXPECT_NEAR(out_ct(0),  1.0f, 1e-6f);
	EXPECT_NEAR(out_ct(1),  0.0f, 1e-6f);
	EXPECT_NEAR(out_ct(2), -1.0f, 1e-6f);
	for (int i = 0; i < 3; ++i)
		EXPECT_NEAR(out_rt(i), out_ct(i), 1e-6f);
}
