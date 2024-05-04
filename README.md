# NumHalide
[NumHalide](https://github.com/soufianekhiat/NumHalide) is header only, it's an attempt to have a feature parity with numpy using Halide for a Halide::Func and a Halide::Buffer<>.

2 projects:
* NumHalide: header only
* NumHalide_Examples: use case, currently a naive Visual Studio code, will change for a [sharpmake](https://github.com/ubisoft/sharpmake) or/and [CMake](https://github.com/Kitware/CMake) settings.

The point of the project is only simply having the syntaxic sugar of Numpy available for Halide user. It will allow us to have the power of Halide!

Let's take a simple example:
```cpp
Func xs = func::linspace(Float(32), 0.0f, 1.0f, width, "xs");
Func ys = func::linspace(Float(32), 0.0f, 1.0f, width, "ys");
std::vector<Func> ids = numhalide::func::meshgrid(Float(32), { xs, ys }, "meshgrid");
Func x = ids[0];
Func y = ids[1];

Var u, v, c;

Expr cx = 2.0f * (x(u, v) - 0.5f);
Expr cy = 2.0f * (y(u, v) - 0.5f);

Expr out = exp(-(cx * cx + cy * cy) / 0.25f);

Expr to8bits = cast(UInt(8), round(pow(clamp(out, 0.0f, 1.0f), 1.0f / 2.2f) * 255.0f));

Func result(UInt(8), 3, "image");
result(c, u, v) = select(c < 3, to8bits, Internal::make_const(UInt(8), 255));
```
That will generate a gaussian centered image in 8 bits. But what matter is the how, Halide did his magic and will produce an elegant code:
```cpp
image[(((image.stride.1 * t145) + t147) + image.s0.v38.rebased)] =
    select(((image.min.0 + image.s0.v38.rebased) < 3),
        uint8(round((pow_f32(max(min(exp_f32((((((t140 * 0.003914f) + -1.000000f) * ((t140 * 0.003914f) + -1.000000f)) + (((t146 * 0.003914f) + -1.000000f) * ((t146 * 0.003914f) + -1.000000f))) * -4.000000f)), 1.000000f), 0.000000f), 0.454545f) * 255.000000f))), (uint8)255)
```
Without the need of storing `xs`, `ys`, `ids`, `cx`, `cy`, ... everything done properly. Note with get this with a default scheduling of `compute_root` or with `Adams2019` on CPU.

Incentivise development:

[<img src="https://c5.patreon.com/external/logo/become_a_patron_button@2x.png" alt="Become a Patron" width="150"/>](https://www.patreon.com/SoufianeKHIAT)

https://www.patreon.com/SoufianeKHIAT

No scheduling is provided. A final design is to split the function and scheduling for a given platform is still open to change.

The NdArray is equivalent to Halide::Buffer<>.

Inspired by [NumCpp](https://github.com/dpilger26/NumCpp).
Here a modified code of [NumCpp](https://github.com/dpilger26/NumCpp).

### CONTAINERS

// TODO

| NumPy | NumCpp | NumHalide |
| --- | --- | --- |
| a = np.array([[1, 2], [3, 4], [5, 6]]) | nc::NdArray<int> a = { {1, 2}, {3, 4}, {5, 6} } |  |
| a.reshape([2, 3]) | a.reshape(2, 3) | reshape(a, {3, 2}, {2, 3}) |
| a.astype(np.double) | a.astype<double>() | cast<double>(a) |

### INITIALIZERS

// TODO

| NumPy | NumCpp | NumHalide |
| --- | --- | --- |
| np.linspace(1, 10, 5) | nc::linspace<dtype>(1, 10, 5) | // Geneare 2D array, the size is implied at the realize and set_estimate.<br/>linspace(Type, 1, 10, 5) |
| np.arange(3, 7) | nc::arange<dtype>(3, 7) | // Geneare 2D array, the size is implied at the realize and set_estimate.<br/>arange(Type, 3, 7) |
| np.eye(4) | nc::eye<dtype>(4) |  |
| np.zeros([3, 4]) | nc::zeros<dtype>(3, 4) | // Geneare 2D array, the size is implied at the realize and set_estimate.<br/>zeros(Type, 2) |
|  | nc::NdArray<dtype>(3, 4) a = 0 |  |
| np.ones([3, 4]) | nc::ones<dtype>(3, 4) | // Geneare 2D array, the size is implied at the realize and set_estimate.<br/>ones(Type, 2) |
|  | nc::NdArray<dtype>(3, 4) a = 1 |  |
| np.nans([3, 4]) | nc::nans(3, 4) |  |
|  | nc::NdArray<double>(3, 4) a = nc::constants::nan |  |
| np.empty([3, 4]) | nc::empty<dtype>(3, 4) | // Geneare 2D array, the size is implied at the realize and set_estimate.<br/>empty(Type, 2) |
|  | nc::NdArray<dtype>(3, 4) a |  |

### 

### SLICING/BROADCASTING

// TODO

| NumPy | NumCpp | NumHalide |
| --- | --- | --- |
| a[2, 3] | a(2, 3) |  |
| a[2:5, 5:8] | a(nc::Slice(2, 5), nc::Slice(5, 8)) |  |
|  | a({2, 5}, {5, 8}) |  |
| a[:, 7] | a(a.rSlice(), 7) |  |
| a[a > 5] | a[a > 5] |  |
| a[a > 5] = 0 | a.putMask(a > 5, 0) |  |

### 

### RANDOM

// TODO

The random module provides simple ways to create random arrays.

| NumPy | NumCpp | NumHalide |
| --- | --- | --- |
| np.random.seed(666) | nc::random::seed(666) |  |
| np.random.randn(3, 4) | nc::random::randN<double>(nc::Shape(3, 4)) |  |
|  | nc::random::randN<double>({3, 4}) |  |
| np.random.randint(0, 10, [3, 4]) | nc::random::randInt<int>(nc::Shape(3, 4), 0, 10) |  |
|  | nc::random::randInt<int>({3, 4}, 0, 10) |  |
| np.random.rand(3, 4) | nc::random::rand<double>(nc::Shape(3,4)) |  |
|  | nc::random::rand<double>({3, 4}) |  |
| np.random.choice(a, 3) | nc::random::choice(a, 3) |  |

### 

### CONCATENATION

// TODO

| NumPy | NumCpp | NumHalide |
| --- | --- | --- |
| np.stack([a, b, c], axis=0) | nc::stack({a, b, c}, nc::Axis::ROW) |  |
| np.vstack([a, b, c]) | nc::vstack({a, b, c}) |  |
| np.hstack([a, b, c]) | nc::hstack({a, b, c}) |  |
| np.append(a, b, axis=1) | nc::append(a, b, nc::Axis::COL) |  |

### 

### DIAGONAL, TRIANGULAR, AND FLIP

// TODO

| NumPy | NumCpp | NumHalide |
| --- | --- | --- |
| np.diagonal(a) | nc::diagonal(a) |  |
| np.triu(a) | nc::triu(a) |  |
| np.tril(a) | nc::tril(a) |  |
| np.flip(a, axis=0) | nc::flip(a, nc::Axis::ROW) |  |
| np.flipud(a) | nc::flipud(a) |  |
| np.fliplr(a) | nc::fliplr(a) |  |

### 

### ITERATION

// TODO

| NumPy | NumCpp | NumHalide |
| --- | --- | --- |
| for value in a | for(auto it = a.begin(); it < a.end(); ++it) | "a(_)" |
|  | for(auto& value : a) |  |

### 

### LOGICAL

// TODO

| NumPy | NumCpp | NumHalide |
| --- | --- | --- |
| np.where(a > 5, a, b) | nc::where(a > 5, a, b) |  |
| np.any(a) | nc::any(a) |  |
| np.all(a) | nc::all(a) |  |
| np.logical_and(a, b) | nc::logical_and(a, b) |  |
| np.logical_or(a, b) | nc::logical_or(a, b) |  |
| np.isclose(a, b) | nc::isclose(a, b) |  |
| np.allclose(a, b) | nc::allclose(a, b) |  |

### 

### COMPARISONS

// TODO

| NumPy | NumCpp | NumHalide |
| --- | --- | --- |
| np.equal(a, b) | nc::equal(a, b) | a( _ ) == b( _ ) |
|  | a == b |  |
| np.not_equal(a, b) | nc::not_equal(a, b) | a( _ ) != b( _ ) |
|  | a != b |  |
| rows, cols = np.nonzero(a) | auto [rows, cols] = nc::nonzero(a) |  |

### 

### MINIMUM, MAXIMUM, SORTING

// TODO

| NumPy | NumCpp | NumHalide |
| --- | --- | --- |
| np.min(a) | nc::min(a) | RDom r( ... );<br/> minimum(a(r)) |
| np.max(a) | nc::max(a) | RDom r( ... );<br/> maximum(a(r)) |
| np.argmin(a) | nc::argmin(a) | RDom r( ... );<br/> argmin(a(r)) |
| np.argmax(a) | nc::argmax(a) | RDom r( ... );<br/> argmax(a(r)) |
| np.sort(a, axis=0) | nc::sort(a, nc::Axis::ROW) |  |
| np.argsort(a, axis=1) | nc::argsort(a, nc::Axis::COL) |  |
| np.unique(a) | nc::unique(a) |  |
| np.setdiff1d(a, b) | nc::setdiff1d(a, b) |  |
| np.diff(a) | nc::diff(a) |  |

### 

### REDUCERS

// TODO

| NumPy | NumCpp | NumHalide |
| --- | --- | --- |
| np.sum(a) | nc::sum(a) |  |
| np.sum(a, axis=0) | nc::sum(a, nc::Axis::ROW) |  |
| np.prod(a) | nc::prod(a) |  |
| np.prod(a, axis=0) | nc::prod(a, nc::Axis::ROW) |  |
| np.mean(a) | nc::mean(a) |  |
| np.mean(a, axis=0) | nc::mean(a, nc::Axis::ROW) |  |
| np.count_nonzero(a) | nc::count_nonzero(a) |  |
| np.count_nonzero(a, axis=0) | nc::count_nonzero(a, nc::Axis::ROW) |  |

### 

### I/O

// TODO

| NumPy | NumCpp | NumHalide |
| --- | --- | --- |
| print(a) | a.print() |  |
|  | std::cout << a |  |
| a.tofile(filename, sep=’\n’) | a.tofile(filename, '\n') |  |
| np.fromfile(filename, sep=’\n’) | nc::fromfile<dtype>(filename, '\n') |  |
| np.dump(a, filename) | nc::dump(a, filename) |  |
| np.load(filename) | nc::load<dtype>(filename) |  |

### 

### MATHEMATICAL FUNCTIONS

// TODO

### 

### BASIC FUNCTIONS

// TODO

| NumPy | NumCpp | NumHalide |
| --- | --- | --- |
| np.abs(a) | nc::abs(a) | abs(a(_)) |
| np.sign(a) | nc::sign(a) |  |
| np.remainder(a, b) | nc::remainder(a, b) |  |
| np.clip(a, 3, 8) | nc::clip(a, 3, 8) | clamp(a(_), 3, 8) |
| np.interp(x, xp, fp) | nc::interp(x, xp, fp) |  |

### 

### EXPONENTIAL FUNCTIONS

// TODO

| NumPy | NumCpp | NumHalide |
| --- | --- | --- |
| np.exp(a) | nc::exp(a) | exp(a(_)) |
| np.expm1(a) | nc::expm1(a) |  |
| np.log(a) | nc::log(a) | log(a(_)) |
| np.log1p(a) | nc::log1p(a) |  |

### 

### POWER FUNCTIONS

// TODO

| NumPy | NumCpp | NumHalide |
| --- | --- | --- |
| np.power(a, 4) | nc::power(a, 4) | pow(a(_), 4) |
| np.sqrt(a) | nc::sqrt(a) |  |
| np.square(a) | nc::square(a) |  |
| np.cbrt(a) | nc::cbrt(a) |  |

### 

### TRIGONOMETRIC FUNCTIONS

| NumPy | NumCpp | NumHalide |
| --- | --- | --- |
| np.sin(a) | nc::sin(a) | sin(a(_)) |
| np.cos(a) | nc::cos(a) | cos(a(_)) |
| np.tan(a) | nc::tan(a) | tan(a(_)) |

### 

### HYPERBOLIC FUNCTIONS

| NumPy | NumCpp | NumHalide |
| --- | --- | --- |
| np.sinh(a) | nc::sinh(a) | sinh(a(_)) |
| np.cosh(a) | nc::cosh(a) | cosh(a(_)) |
| np.tanh(a) | nc::tanh(a) | tanh(a(_)) |

### 

### CLASSIFICATION FUNCTIONS

| NumPy | NumCpp | NumHalide |
| --- | --- | --- |
| np.isnan(a) | nc::isnan(a) |  |
| np.isinf(a) | nc::isinf(a) |  |

### 

### LINEAR ALGEBRA

// TODO

| NumPy | NumCpp | NumHalide |
| --- | --- | --- |
| np.linalg.norm(a) | nc::norm(a) |  |
| np.dot(a, b) | nc::dot(a, b) |  |
| np.linalg.det(a) | nc::linalg::det(a) |  |
| np.linalg.inv(a) | nc::linalg::inv(a) |  |
| np.linalg.lstsq(a, b) | nc::linalg::lstsq(a, b) |  |
| np.linalg.matrix_power(a, 3) | nc::linalg::matrix_power(a, 3) |  |
| Np.linalg.multi_dot(a, b, c) | nc::linalg::multi_dot({a, b, c}) |  |
| np.linalg.svd(a) | nc::linalg::svd(a) |  |
