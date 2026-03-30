# Automatic Differentiation

NumHalide provides two pure-C++ reverse-mode AD systems. Neither uses Halide JIT.

---

## Scalar AD — `DVar`

`#include "autodiff.h"`

Single-variable scalar reverse-mode AD via a thread-local tape.

### Usage

```cpp
dtape_reset();          // Clear the tape before each computation
DVar x(1.2f);
DVar y(0.5f);
DVar z = dexp(x * x) + dtanh(y);
z.backward();           // Propagate gradients from z
float dz_dx = x.grad(); // ∂z/∂x = 2x * exp(x²)
float dz_dy = y.grad(); // ∂z/∂y = 1 - tanh²(y)
```

### API

| Item | Notes |
|---|---|
| `dtape_reset()` | Must be called before each new computation |
| `DVar(float)` | Create a leaf variable |
| `x.backward()` | Back-propagate; initializes grad of `x` to 1 |
| `x.grad()` | Accumulated gradient (float) |
| `+`, `-`, `*`, `/`, unary `-` | Operator overloads |
| `dexp(x)` | e^x |
| `dlog(x)` | ln x |
| `dsin(x)` | sin x |
| `dcos(x)` | cos x |
| `dsqrt(x)` | √x |
| `dpow(x, y)` | x^y |
| `dtanh(x)` | tanh x |
| `dabs(x)` | |x| (subgradient 0 at x=0) |

---

## Tensor AD — `Tensor` and `TVar`

`#include "autodiff_tensor.h"`

Full N-dimensional (up to 8D) reverse-mode AD. Row-major flat storage.

### Tensor — Plain Value Type

No gradient tracking. Used to hold data and pass to `TVar`.

```cpp
Tensor s(3.0f);                             // scalar
Tensor v({1.0f, 2.0f, 3.0f});              // 1D
Tensor M({{1.f,2.f},{3.f,4.f}});           // 2D
Tensor T = Tensor::zeros({2, 3, 4});        // ND
Tensor I = Tensor::eye(3);                  // identity
```

| Method | Notes |
|---|---|
| `t.rank()` | Number of dimensions |
| `t.dim(i)` | Size of dimension `i` |
| `t.size()` | Total element count |
| `t.flat(k)` | Element at flat index `k` |
| `t(i)`, `t(i,j)`, `t(b,i,j)`, `t(n,c,h,w)` | Index accessors |
| `t.is_scalar()` | True if rank == 0 |
| `t.item()` | Scalar value (rank must be 0) |
| `Tensor::zeros(shape)` | — |
| `Tensor::ones(shape)` | — |
| `Tensor::eye(n)` | n×n identity |
| `Tensor::make_strides(shape)` | C-order strides |
| `t.matmul(b)` | 2D matrix multiply |
| `t.batch_matmul(b)` | `{BS,M,K} @ {BS,K,N}` |
| `t.transpose()` | 2D transpose |
| `t.sum_axis(axis)` | Reduce along one axis (ND) |
| `t.norm()` | L2 norm |
| `t.dot(b)` | 1D dot product |
| `t.trace()` | 2D trace |

### TVar — Differentiable Wrapper

```cpp
ttape_reset();                          // Clear tape before computation
TVar W(Tensor({{1.f,0.f},{0.f,1.f}})); // 2D leaf
TVar x(std::vector<float>{3.f, 4.f});  // 1D leaf — use explicit vector<float>
TVar loss = tnorm(tmatmul(W, x));
loss.backward();
Tensor dW = W.grad();
Tensor dx = x.grad();
```

> **Construction note:** `TVar({1.f, 2.f})` is ambiguous (matches both `vector<float>` and `Tensor` constructor). Always use `TVar(std::vector<float>{1.f, 2.f})` for 1D leaves.

| Item | Notes |
|---|---|
| `ttape_reset()` | Must be called before each new computation |
| `TVar(float)` | Scalar leaf |
| `TVar(std::vector<float>)` | 1D leaf |
| `TVar({{...}, {...}})` | 2D leaf from nested initializer_list |
| `TVar(Tensor)` | ND leaf |
| `v.backward()` | Back-propagate; `v` must be scalar |
| `v.grad()` | Accumulated gradient as `Tensor` |
| `v.val()` | Forward value as `Tensor` |
| `+`, `-`, `*`, `/`, unary `-` | Elementwise; scalar broadcast supported |

### Free Functions

| Category | Function | NumPy analogue |
|---|---|---|
| **Matrix** | `tmatmul(A, B)` | `A @ B` |
| | `ttranspose(A)` | `A.T` |
| | `tbatch_matmul(A, B)` | `np.matmul(A, B)` batched |
| | `ttrace(A)` | `np.trace(A)` |
| | `tdot(a, b)` | `np.dot(a, b)` |
| **Reduction** | `tsum(x)` | `np.sum(x)` |
| | `tsum(x, axis)` | `np.sum(x, axis)` |
| | `tmean(x)` | `np.mean(x)` |
| | `tnorm(x)` | `np.linalg.norm(x)` |
| | `tfrobenius(A)` | `np.linalg.norm(A, 'fro')` |
| | `tfrobenius_sq(A)` | `np.linalg.norm(A, 'fro')**2` |
| **Reshape** | `treshape(x, shape)` | `x.reshape(shape)` |
| | `tnormalize(x)` | `x / np.linalg.norm(x)` |
| **Activation** | `trelu(x)` | `np.maximum(x, 0)` |
| | `tsigmoid(x)` | `1 / (1 + np.exp(-x))` |
| | `ttanh(x)` | `np.tanh(x)` |
| **Elementwise** | `texp(x)` | `np.exp(x)` |
| | `tlog(x)` | `np.log(x)` |
| | `tsin(x)` | `np.sin(x)` |
| | `tcos(x)` | `np.cos(x)` |
| | `tsqrt(x)` | `np.sqrt(x)` |
| | `tpow(x, y)` | `np.power(x, y)` |
| | `tabs(x)` | `np.abs(x)` |

### Gradient Examples

```cpp
// Least squares: d||Ax - b||² / dx = 2 A^T (Ax - b)
ttape_reset();
TVar A(Av), x(xv), b(bv);
tsum((tmatmul(A, x) - b) * (tmatmul(A, x) - b)).backward();
Tensor dx = x.grad();

// Cross-entropy: d/dz [log(sum(exp(z))) - z[y]] = softmax(z) - one_hot(y)
ttape_reset();
TVar z(zv), y_onehot(yv);
(tlog(tsum(texp(z))) - tdot(y_onehot, z)).backward();
Tensor dz = z.grad();

// Batch matmul: {BS,M,K} @ {BS,K,N}
ttape_reset();
TVar A(Tensor({BS,M,K}, ad)), B(Tensor({BS,K,N}, bd));
tsum(tbatch_matmul(A, B)).backward();
Tensor dA = A.grad(), dB = B.grad();
```
