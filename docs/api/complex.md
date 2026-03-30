# Complex Numbers

`#include "complex_type.h"`

NumPy uses `np.complex64` natively in arrays. NumHalide separates real and imaginary parts into two `Halide::Buffer<float>` (via `ComplexBuffer`) to avoid ABI issues with Halide's type system. FFT functions use Tuple-returning Funcs for the same reason.

## complex_f32 — Value Type

Plain C++ struct for scalar complex arithmetic. No Halide involvement.

| Method / Function | NumPy / Python | Notes |
|---|---|---|
| `complex_f32(re, im)` | `complex(re, im)` | — |
| `z.re`, `z.im` | `z.real`, `z.imag` | Public fields |
| `z + w`, `z - w`, `z * w`, `z / w` | Operators | — |
| `z.abs()` | `abs(z)` | Modulus |
| `z.phase()` | `cmath.phase(z)` | atan2(im, re) |
| `z.conj()` | `z.conjugate()` | — |
| `complex_from_polar(r, theta)` | `cmath.rect(r, theta)` | `r * exp(i*theta)` |

## ComplexBuffer — Re/Im Buffer Pair

`ComplexBuffer` stores a paired `Halide::Buffer<float>` for real and imaginary parts.

| Method / Function | NumPy | Notes |
|---|---|---|
| `ComplexBuffer cb(N)` | `np.zeros(N, dtype=np.complex64)` | Zero-initialized |
| `cb.set(i, z)` | `arr[i] = z` | — |
| `cb(i)` | `arr[i]` | Returns `complex_f32` |
| `cb.size()` | `len(arr)` | — |
| `complex_buf_add(a, b)` | `a + b` | Elementwise |
| `complex_buf_mul(a, b)` | `a * b` | Elementwise |
| `complex_buf_abs(cb)` | `np.abs(arr)` | Returns `Buffer<float>` |
| `complex_buf_phase(cb)` | `np.angle(arr)` | Returns `Buffer<float>` |

## Halide Pipeline Bridge

Connect a `ComplexBuffer` to a Halide FFT pipeline and back.

```cpp
// ComplexBuffer → Func (for use in FFT pipelines)
Halide::Func f = complex_buf_to_func(cb, "signal");

// Run FFT
Halide::Func F = fft(f, N, "fft_out");

// Func → ComplexBuffer (realize and collect)
ComplexBuffer result = complex_func_to_buf(F, N);
```

**`complex_buf_to_func`** copies re/im data into two `Halide::Buffer<float>` and wraps them as a Tuple-returning Func.

**`complex_func_to_buf`** realizes the Func (calling Halide JIT) and copies the result into a `ComplexBuffer`.
