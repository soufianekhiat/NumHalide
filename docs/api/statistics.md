# Statistics

## Variance and Standard Deviation

`#include "stats.h"` — functions in the `stats::` nested namespace.

| Function | NumPy | Notes |
|---|---|---|
| `stats::var(f, shape)` | `np.var(a)` | Full reduction |
| `stats::var(f, shape, axis)` | `np.var(a, axis)` | — |
| `stats::var(f, shape, axis, ddof)` | `np.var(a, ddof=1)` | `ddof=1` = Bessel correction |
| `stats::std(f, shape)` | `np.std(a)` | — |
| `stats::std(f, shape, axis)` | `np.std(a, axis)` | — |
| `stats::std(f, shape, axis, ddof)` | `np.std(a, ddof=1)` | — |

## Extended Statistics

`#include "stats_ext.h"` — functions in the `stats::` nested namespace.

| Function | NumPy | Notes |
|---|---|---|
| `stats::median(f, shape)` | `np.median(a)` | Sorts internally; works on flattened input |
| `stats::ptp(f, shape)` | `np.ptp(a)` | Peak-to-peak = max − min |
| `stats::average(f, w, shape)` | `np.average(a, weights=w)` | Weighted mean; `w` is a Func |
| `stats::histogram(f, shape, bins, min, max)` | `np.histogram(a, bins, range)` | Returns bin-count Func (float) |
| `stats::digitize(f, bins, shape, n_bins)` | `np.digitize(a, bins)` | `bins` is a Func of size `n_bins` |
