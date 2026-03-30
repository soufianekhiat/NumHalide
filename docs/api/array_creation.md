# Array Creation

## Factory Functions

`#include "factory_func.h"`

| Function | NumPy | Notes |
|---|---|---|
| `zeros(Type, shape)` | `np.zeros(shape, dtype)` | Takes `Halide::Type` as first arg |
| `ones(Type, shape)` | `np.ones(shape, dtype)` | — |
| `full(Type, shape, value)` | `np.full(shape, value, dtype)` | — |
| `linspace(Type, start, stop, n)` | `np.linspace(start, stop, n)` | Returns `Func`, not array |
| `arange(Type, start, stop, step=1)` | `np.arange(start, stop, step)` | Size must be known at compile time |
| `eye(Type, n)` | `np.eye(n, dtype)` | Square only |
| `meshgrid(Type, {f0, f1, ...})` | `np.meshgrid(x, y, ...)` | Takes list of 1D Funcs; returns `vector<Func>` |
| `zeros_like(f, shape)` | `np.zeros_like(a)` | — |
| `ones_like(f, shape)` | `np.ones_like(a)` | — |
| `full_like(f, shape, value)` | `np.full_like(a, value)` | — |

```cpp
auto z = zeros(Halide::Float(32), {4, 4});
auto r = linspace(Halide::Float(32), 0.0f, 1.0f, 100);
auto I = eye(Halide::Float(32), 3);
```

## Random Generation

`#include "factory_func.h"` (basic), `#include "random_ext.h"` (extended)

| Function | NumPy | Notes |
|---|---|---|
| `rand_uniform(Type, shape, seed)` | `np.random.rand(...)` | Seed is required |
| `rand_normal(Type, shape, mean, stddev)` | `np.random.randn(...)` | Explicit mean/stddev, seed optional |
| `rand_int(Type, shape, lo, hi, seed)` | `np.random.randint(lo, hi, shape)` | `[lo, hi)` half-open |
| `rand_exponential(Type, shape, lambda, seed)` | `np.random.exponential(1/lambda, shape)` | Takes rate λ, not scale |
| `rand_bernoulli(Type, shape, p, seed)` | `np.random.binomial(1, p, shape)` | Returns float 0.0 or 1.0 |
| `rand_choice(Type, shape, n, seed)` | `np.random.randint(0, n, shape)` | Uniform integer in `[0, n)` |

```cpp
auto u = rand_uniform(Halide::Float(32), {512, 512}, 42);
auto g = rand_normal(Halide::Float(32), {1024}, 0.0f, 1.0f, 7);
```
