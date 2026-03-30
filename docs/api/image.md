# Image Processing

## Interpolation and Resampling

`#include "interp.h"`

| Function | NumPy / OpenCV / SciPy | Notes |
|---|---|---|
| `interp1d_uniform(y, shape, scale)` | `np.interp(x_new, x, y)` | Uniform source grid; `scale` = x_new/x_old ratio |
| `resize_bilinear(f, shape, out_h, out_w)` | `cv2.resize(..., INTER_LINEAR)` | — |
| `resize_nearest(f, shape, out_h, out_w)` | `cv2.resize(..., INTER_NEAREST)` | — |
| `zoom(f, shape, factor)` | `scipy.ndimage.zoom(a, factor)` | Uniform scale factor |
| `map_coordinates(f, shape, cx, cy)` | `scipy.ndimage.map_coordinates` | `cx`, `cy` are Funcs of output coords |

## Morphological Operations

`#include "morphology.h"`

| Function | SciPy / OpenCV | Notes |
|---|---|---|
| `dilate(f, shape, k)` | `ndimage.maximum_filter(a, k)` | Square structuring element of radius `k` |
| `erode(f, shape, k)` | `ndimage.minimum_filter(a, k)` | — |
| `morph_open(f, shape, k)` | `ndimage.binary_opening` | Erode then dilate |
| `morph_close(f, shape, k)` | `ndimage.binary_closing` | Dilate then erode |
| `morph_gradient(f, shape, k)` | — | Dilate − Erode |
| `top_hat(f, shape, k)` | — | Input − Open |

## Color Space Conversions

`#include "color.h"`

Input/output convention: `f(x, y, c)` where `c ∈ {0,1,2}` is the channel.

| Function | Reference | Notes |
|---|---|---|
| `rgb_to_gray(f, shape)` | BT.601 luma | Returns 2D (single-channel) |
| `gray_to_rgb(f, shape)` | — | Replicates to 3 channels |
| `rgb_to_hsv(f, shape)` | — | H ∈ [0,1], S ∈ [0,1], V ∈ [0,1] |
| `hsv_to_rgb(f, shape)` | — | — |
| `rgb_to_yuv(f, shape)` | BT.601 | Y ∈ [0,1], U/V ∈ [−0.5,0.5] |
| `yuv_to_rgb(f, shape)` | BT.601 | — |

## Histogram and LUT

`#include "histogram.h"`

| Function | NumPy / OpenCV | Notes |
|---|---|---|
| `histogram_1d(f, shape, bins, min, max)` | `np.histogram(a, bins, range)` | Returns float count Func |
| `cumulative_histogram(f, shape, bins, min, max)` | — | CDF of histogram |
| `histogram_equalize(f, shape, bins)` | `cv2.equalizeHist` | Assumes input in [0,1] |
| `apply_lut(f, lut, shape)` | — | `lut` is a 1D Func |
| `gamma_correct(f, shape, gamma)` | `cv2.LUT` with gamma | `out = clamp(in,0,1)^gamma` |

## Thresholding

`#include "threshold.h"`

| Function | OpenCV | Notes |
|---|---|---|
| `threshold_binary(f, shape, thresh)` | `THRESH_BINARY` | 1 if above, 0 if below |
| `threshold_trunc(f, shape, thresh)` | `THRESH_TRUNC` | Clamp to thresh from above |
| `threshold_tozero(f, shape, thresh)` | `THRESH_TOZERO` | Zero below threshold |
| `threshold_otsu(f, shape, bins)` | `THRESH_OTSU` | Threshold computed from histogram |
| `threshold_adaptive(f, shape, block_size)` | `ADAPTIVE_THRESH_MEAN_C` | Local mean threshold |

## Gradient and Differential Operators

`#include "gradient.h"`

| Function | NumPy | Notes |
|---|---|---|
| `gradient_1d(f, shape, axis)` | `np.gradient(a, axis=axis)` | Central differences, edge-clamped |
| `gradient_2d(f, shape)` | `np.gradient(a)` | Returns `{dx, dy}` pair |
| `laplacian(f, shape)` | — | Discrete 5-point Laplacian |
| `divergence(fx, fy, shape)` | — | `∂fx/∂x + ∂fy/∂y` |

## Stencil Operations

`#include "stencil.h"`

| Function | Description | Notes |
|---|---|---|
| `stencil_apply(f, shape, weights, offsets_x, offsets_y, n)` | Weighted neighbor sum | `n` = number of stencil points |
| `jacobi_step(f, shape)` | One Jacobi iteration | 5-point stencil, uniform coefficients |
| `heat_diffusion_step(f, shape, dt, alpha)` | Explicit heat equation step | `u_new = u + dt*alpha*laplacian(u)` |
