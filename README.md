# image-array-and-histogram

[![CI](https://github.com/rishi-chauhan/my-packages/actions/workflows/ci.yml/badge.svg)](https://github.com/rishi-chauhan/my-packages/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/image-array-and-histogram.svg)](https://pypi.org/project/image-array-and-histogram/)
[![Python versions](https://img.shields.io/pypi/pyversions/image-array-and-histogram.svg)](https://pypi.org/project/image-array-and-histogram/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Fast image histograms, exposure statistics and histogram comparison for NumPy and Pillow,
with a command-line tool.

If you just need a 256-bin histogram of one image, `img.convert("L").histogram()` is enough.
This library adds what the built-ins don't: per-channel and 16-bit/float histograms, exposure
statistics (percentiles, clipping, entropy), histogram comparison, streaming accumulation over
whole datasets, and the `imghist` CLI. It depends only on NumPy and Pillow, and the hot paths
are vectorised (`np.bincount` and Pillow's C code, never Python loops).

## Installation

```sh
pip install image-array-and-histogram
# with plotting support for `imghist --plot`:
pip install "image-array-and-histogram[plot]"
```

Requires Python 3.11+ (tested on 3.11–3.15), NumPy 1.23.2+ and Pillow 13+. The minimum
Pillow version is the first release without publicly known security vulnerabilities, since
this library opens arbitrary image files. The package ships type hints.

## Quick start

```python
from PIL import Image

from image_array_and_histogram import HistAccumulator, compare_hist, get_hist, hist_stats

img = Image.open("photo.jpg")

# Histograms: 256 gray bins, or one row per RGB channel
gray = get_hist(img)  # ndarray, shape (256,), int64
rgb = get_hist(img, channels="rgb")  # shape (3, 256)
coarse = get_hist(img, bins=16, density=True)  # 16 bins summing to 1.0

# Exposure statistics, computed from the histogram
stats = hist_stats(img)
print(f"mean {stats.mean:.1f}, median {stats.median}, p99 {stats.percentiles[99]}")
print(f"blown highlights: {stats.clipped_high:.2%}")

# Compare two images (1.0 = identical for intersection)
other = img.rotate(90)
print(compare_hist(get_hist(img), get_hist(other), method="intersection"))

# Accumulate one histogram over many images without holding them in memory
acc = HistAccumulator(channels="rgb")
for path in ["photo.jpg", "photo.jpg"]:
    with Image.open(path) as im:
        acc.update(im)
print(acc.n_images, acc.stats()["r"].mean)
```

16-bit images keep their full precision:

```python
import numpy as np

from image_array_and_histogram import get_hist, hist_stats

deep = np.random.default_rng(0).integers(0, 65536, size=(480, 640), dtype=np.uint16)
counts = get_hist(deep, bins=1024)  # range defaults to (0, 65536) for uint16
print(hist_stats(deep, bins=65536).median)
```

## Command line

```bash
imghist stats photo.jpg
imghist stats ./dataset -r --channels rgb -j 0
imghist stats ./dataset -r --format csv --per-file -o report.csv
imghist compare day.jpg night.jpg --method bhattacharyya
pip install "image-array-and-histogram[plot]" && imghist stats ./dataset -r --plot hist.png
```

Example output:

```text
imghist: 1,204 images (3 skipped) · channel: gray · 256 bins

count     14,412,000,000
mean      118.42    std 61.07
min       0         max 255
median    112       p1 3  p5 14  p25 69  p75 167  p95 228  p99 249
entropy   7.62 bits
clipped   low 0.41%  high 1.87%

▁▁▂▃▄▅▆▇██▇▆▅▅▄▄▃▃▃▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂
0                                                            255
```

`imghist stats PATH [PATH ...]` options:

| Option | Default | Meaning |
|---|---|---|
| `-r, --recursive` | off | recurse into directories |
| `--ext` | `.png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp,.gif` | extensions to include (case-insensitive) |
| `--channels` | `gray` | `gray` or `rgb` |
| `--bins` | `256` | number of bins |
| `--range LO HI` | dtype default | histogram range; required for float images and to mix 8/16-bit images |
| `--per-file` | off | also report each file |
| `--format` | `text` | `text`, `json` or `csv` |
| `--width` | `64` | width of the text histogram |
| `--plot FILE` | none | save a plot (needs the `[plot]` extra) |
| `-j, --jobs` | `1` | worker processes; `0` = one per CPU |
| `--strict` | off | stop at the first unreadable file instead of skipping it |
| `-o, --output FILE` | stdout | write the report to a file |

`imghist compare A B` accepts files or directories (a directory is accumulated into one
histogram) and the same discovery options, plus `--method` and `--format text|json`.

Hidden files and directories are skipped. Unreadable files are reported on stderr and skipped.
Exit codes: `0` success, `1` no images found / none readable / `--strict` failure, `2` usage
error. `python -m image_array_and_histogram` works too.

## API reference

All functions accept a `PIL.Image.Image` or an array: `(H, W)` grayscale, or `(H, W, 3)` /
`(H, W, 4)` RGB(A) (alpha is ignored). 8-bit, 16-bit, other integer and float data are
supported. Booleans are treated as 0/255. NaN/inf values raise `ValueError`.

### `get_hist(image, *, channels="gray", bins=256, value_range=None, density=False)`

Returns int64 counts (float64 with `density=True`) of shape `(bins,)` for `"gray"` or
`(3, bins)` for `"rgb"` (rows R, G, B).

- `value_range` defaults by dtype: uint8 `(0, 256)`, uint16 `(0, 65536)`, other integers
  `(min, max + 1)`, floats `(0, 1)` if all data is in [0, 1], otherwise `(min, max)`.
- Bin edges are `np.linspace(low, high, bins + 1)`. Bins are half-open except the last, which
  includes `high`, and values outside the range are ignored (same as `np.histogram`).
- Grayscale conversion uses ITU-R 601-2 luma (`0.299 R + 0.587 G + 0.114 B`), via Pillow for
  8-bit data, so arrays and PIL images give identical results.

### `hist_stats(image, *, channels="gray", bins=256, value_range=None, percentiles=(1, 5, 25, 50, 75, 95, 99))`

Returns a `HistStats` (or `{"r": ..., "g": ..., "b": ...}` for `"rgb"`) with `count`, `mean`,
`std` (population), `min`, `max`, `median`, `percentiles`, `entropy` (bits), `clipped_low` and
`clipped_high` (fraction of pixels in the first and last bin). `HistStats.to_dict()` gives a
JSON-ready dict with percentile keys like `"p5"`.

Statistics are computed from the histogram in O(bins). They are exact for integer data with
bin width 1 (the uint8 default) and approximate to bin resolution otherwise. Percentiles use
the nearest-rank (inverted CDF) method.

### `stats_from_counts(counts, edges, percentiles=...)`

The same statistics for a histogram you already have (`counts` of shape `(bins,)`, `edges` of
shape `(bins + 1,)`).

### `compare_hist(h1, h2, *, method="intersection")`

Compares two histograms of the same shape and returns a float. Histograms are normalised to
sum to 1 first (except for `correlation`), so images of different sizes compare fairly. For
`(channels, bins)` input the per-channel scores are averaged.

| method | range | identical | definition |
|---|---|---|---|
| `intersection` | [0, 1] | 1 | `sum(min(p, q))` |
| `chi2` | [0, 1] | 0 | `0.5 * sum((p - q)² / (p + q))` |
| `bhattacharyya` | [0, 1] | 0 | `sqrt(1 - sum(sqrt(p * q)))` (Hellinger form) |
| `correlation` | [-1, 1] | 1 | Pearson r of the raw counts |
| `emd` | [0, bins - 1] | 0 | `sum(abs(cumsum(p) - cumsum(q)))`, Earth Mover's distance in bins |

### `HistAccumulator(*, channels="gray", bins=256, value_range=None)`

Streams images into one histogram: `update(image)`, `merge(other)` (for multiprocessing),
`counts` (read-only), `edges`, `n_images` and `stats(percentiles)`. Without `value_range`, the
range comes from the first image's dtype (uint8 or uint16); float and other integer images
need an explicit `value_range` so bins line up across images.

### `get_image_array(image, *, mode="gray")`

Returns a PIL image as an array: `"gray"` gives `(H, W)`, `"rgb"` gives `(H, W, 3)`, `"keep"`
gives `np.asarray(image)` unchanged. The native bit depth is kept (16-bit images give uint16).

### `array_to_image(arr, width=None, height=None, *, clip=True)`

Builds a PIL image from a 1-D array (with `width` and `height`), an `(H, W)` array (mode `L`, or
lossless `I;16` for uint16) or an `(H, W, 3/4)` array (`RGB`/`RGBA`). Other dtypes are converted
to 8 bits: floats are rounded (not truncated); values outside [0, 255] are clipped with a
`RuntimeWarning` when `clip=True`, or raise `ValueError` when `clip=False`.

## Migrating from 1.x

| 1.x | 2.0 |
|---|---|
| `getImageArray(img)`, `getHist(x)`, `getImageFromArray(a, w, h)` | removed; use `get_image_array`, `get_hist`, `array_to_image` |
| `get_hist(x)` returns a `list` | returns an `np.ndarray`; call `.tolist()` if you need a list |
| `get_hist(x, as_density=True)` | `get_hist(x, density=True)` |
| `get_image_array(img, ensure_grayscale=True)` | `get_image_array(img)` (`mode="gray"`) |
| `get_image_array(img, ensure_grayscale=False)` | `get_image_array(img, mode="keep")` |
| `array_to_image(a, validate_range=False)` | `array_to_image(a, clip=False)` |
| `from image_array_and_histogram.main import ...` | `from image_array_and_histogram import ...` |

Behaviour changes to watch for:

- 16-bit images are no longer reduced to 8 bits: `get_image_array` returns uint16 and
  `get_hist` uses the range `(0, 65536)`. If you relied on 8-bit output, convert first with
  `img.convert("L")`.
- `array_to_image` now accepts `(H, W, 3/4)` arrays (RGB/RGBA) instead of raising, and keeps 2-D
  uint16 arrays as 16-bit images.

## Development

```sh
git clone https://github.com/rishi-chauhan/my-packages.git
cd my-packages
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy
pytest --cov=image_array_and_histogram --cov-fail-under=95
python benchmarks/bench_hist.py  # optional performance check
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License. See [LICENSE](LICENSE).
