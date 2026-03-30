# Math Functions

## Element-wise Operations

`#include "ops.h"`

| Function | NumPy | Notes |
|---|---|---|
| `where(cond, f, g, shape)` | `np.where(cond, a, b)` | All three are Funcs |
| `clip(f, shape, lo, hi)` | `np.clip(a, lo, hi)` | Scalar bounds |
| `astype(f, shape, Type)` | `a.astype(dtype)` | `Type` is `Halide::Type` |
| `sign(f, shape)` | `np.sign(a)` | Returns −1, 0, or 1 as float |
| `square(f, shape)` | `np.square(a)` | — |
| `reciprocal(f, shape)` | `np.reciprocal(a)` | — |

**Halide builtins** (apply inside a Func definition, not to a full array):

| Expression | NumPy |
|---|---|
| `Halide::abs(f(x))` | `np.abs(a)` |
| `Halide::sqrt(f(x))` | `np.sqrt(a)` |
| `Halide::exp(f(x))` | `np.exp(a)` |
| `Halide::log(f(x))` | `np.log(a)` |
| `Halide::pow(f(x), e)` | `np.power(a, e)` |
| `Halide::floor(f(x))` | `np.floor(a)` |
| `Halide::ceil(f(x))` | `np.ceil(a)` |
| `Halide::round(f(x))` | `np.round(a)` |
| `Halide::min(f(x), g(x))` | `np.minimum(a, b)` |
| `Halide::max(f(x), g(x))` | `np.maximum(a, b)` |

## Trigonometry

`#include "trig.h"` · builtins via `Halide::`

| Function | NumPy | Notes |
|---|---|---|
| `Halide::sin(f(x))` | `np.sin(a)` | Inside Func definition |
| `Halide::cos(f(x))` | `np.cos(a)` | — |
| `Halide::tan(f(x))` | `np.tan(a)` | — |
| `Halide::asin(f(x))` | `np.arcsin(a)` | — |
| `Halide::acos(f(x))` | `np.arccos(a)` | — |
| `Halide::atan(f(x))` | `np.arctan(a)` | — |
| `Halide::atan2(y(x), x(x))` | `np.arctan2(y, x)` | — |
| `Halide::sinh(f(x))` | `np.sinh(a)` | — |
| `Halide::cosh(f(x))` | `np.cosh(a)` | — |
| `Halide::tanh(f(x))` | `np.tanh(a)` | — |
| `asinh(f, shape)` | `np.arcsinh(a)` | Full-array wrapper |
| `acosh(f, shape)` | `np.arccosh(a)` | — |
| `atanh(f, shape)` | `np.arctanh(a)` | — |
| `hypot(f, g, shape)` | `np.hypot(a, b)` | √(a²+b²) |
| `degrees(f, shape)` | `np.degrees(a)` | Radians → degrees |
| `radians(f, shape)` | `np.radians(a)` | Degrees → radians |

## Extended Math

`#include "math_ext.h"`

| Function | NumPy | Notes |
|---|---|---|
| `exp2(f, shape)` | `np.exp2(a)` | — |
| `log2(f, shape)` | `np.log2(a)` | — |
| `log10(f, shape)` | `np.log10(a)` | — |
| `expm1(f, shape)` | `np.expm1(a)` | More accurate for small x |
| `log1p(f, shape)` | `np.log1p(a)` | More accurate for small x |
| `cbrt(f, shape)` | `np.cbrt(a)` | — |
| `sinc(f, shape)` | `np.sinc(a)` | Normalized sinc: `sin(πx)/(πx)` |
| `heaviside(f, h0, shape)` | `np.heaviside(a, h0)` | `h0` is the value at 0 |
| `fmod(f, g, shape)` | `np.fmod(a, b)` | Remainder with same sign as dividend |
| `remainder(f, g, shape)` | `np.remainder(a, b)` | Remainder with same sign as divisor |
| `nan_to_num(f, shape)` | `np.nan_to_num(a)` | NaN→0, ±Inf→±max_float |

## Numeric Utilities

`#include "numeric.h"`

| Function | NumPy | Notes |
|---|---|---|
| `logaddexp(f, g, shape)` | `np.logaddexp(a, b)` | `log(exp(a) + exp(b))`, numerically stable |
| `logaddexp2(f, g, shape)` | `np.logaddexp2(a, b)` | Base-2 variant |
| `copysign(f, g, shape)` | `np.copysign(a, b)` | Magnitude of `a`, sign of `b` |
| `signbit(f, shape)` | `np.signbit(a)` | True if negative (including -0) |
| `trapz_1d(f, n)` | `np.trapz(y)` | Trapezoidal integration, unit spacing |
| `i0(f, shape)` | `scipy.special.i0(a)` | Modified Bessel I₀, polynomial approx |
| `correlate1d(f, g, n)` | `np.correlate(a, b)` | 1D cross-correlation, full mode |
