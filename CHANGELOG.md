# Changelog

All notable changes to this project will be documented in this file.

## [2.1.0] - 2026-11-21

No API changes.

### Security

- Require Pillow 13.0.0 or newer. Every earlier Pillow release has publicly listed
  vulnerabilities (for example buffer overflows and decompression-bomb bypasses in the TIFF,
  JPEG, PSD and GIF decoders, fixed in 13.0.0), and `imghist` opens arbitrary image files.
- Build with setuptools 84.1.0 or newer (fixes for CVE-2025-47273, CVE-2026-59890 and
  GHSA-xmj3-9gf7-h52w).

### Changed

- Requires Python 3.11 or newer: Python 3.10 reached end-of-life in October 2026, and Pillow
  13 no longer supports it.
- Added support for Python 3.15; tested on 3.11–3.15.
- Minimum NumPy is now 1.23.2, the first release with Python 3.11 wheels.

## [2.0.0] - 2026-11-21

A major release that turns the package into a histogram and exposure-statistics toolkit. See
"Migrating from 1.x" in the README.

### Added

- `get_hist` options: `channels="rgb"` (one row per channel), `bins`, `value_range`, and
  native 16-bit, other integer and float data. Fast `np.bincount` paths for uint8/uint16 that
  match `np.histogram` exactly.
- `hist_stats`, `stats_from_counts` and the `HistStats` dataclass: count, mean, std, min, max,
  median, percentiles, entropy and clipped-shadow/highlight fractions, computed in O(bins).
- `compare_hist` with `intersection`, `chi2`, `bhattacharyya`, `correlation` and `emd`.
- `HistAccumulator` for streaming one histogram over many images, with `merge()` for
  multiprocessing.
- The `imghist` command-line tool (`imghist stats`, `imghist compare`) for files and folders:
  text, JSON and CSV output, per-file reports, parallel processing (`-j`), and plots with the
  new `[plot]` extra. Also available as `python -m image_array_and_histogram`.
- `array_to_image` builds RGB/RGBA images from `(H, W, 3/4)` arrays and lossless 16-bit
  images from 2-D uint16 arrays.
- `get_image_array(mode="rgb" | "keep")`.

### Changed

- `get_hist` returns an `np.ndarray` (int64, or float64 with `density=True`) instead of a list.
- `get_hist(as_density=...)` is renamed to `density`, matching `np.histogram`.
- `get_image_array(ensure_grayscale=...)` is replaced by `mode="gray" | "rgb" | "keep"`, and
  16-bit/float images keep their native dtype instead of being reduced to 8 bits.
- `array_to_image(validate_range=...)` is renamed to `clip`.

### Removed

- The deprecated camelCase functions `getImageArray`, `getHist` and `getImageFromArray`.
- The `image_array_and_histogram.main` module; import everything from the package root.

## [1.2.0] - 2026-11-21

No API changes.

### Changed

- Requires Python 3.10 or newer (3.8 and 3.9 are end-of-life). Tested on 3.10–3.14.
- Minimum dependencies raised to NumPy 1.23 and Pillow 10.0.
- The package is now typed: ships `py.typed` and passes `mypy --strict`. Signatures use
  `numpy.typing.ArrayLike` / `NDArray`.
- `__version__` is read from the installed package metadata, so it can no longer drift from
  the released version.
- License metadata uses the SPDX expression `MIT`.

### CI

- Test matrix 3.10–3.14 plus a job that tests the minimum NumPy/Pillow versions.
- Lint job runs `ruff check`, `ruff format --check` and `mypy`; tests enforce 95% coverage;
  builds are checked with `twine check`.
- The publish workflow runs the tests first, refuses to publish if the tag doesn't match the
  package version, and no longer skips existing versions silently.

## [1.1.9] - 2026-11-21

### Fixed

- `array_to_image(..., validate_range=False)` now raises `ValueError` for values outside
  [0, 255], as documented, instead of silently clipping them.
- `get_hist` on an RGB(A) array with values above 255 no longer wraps around (300 was
  counted in bin 44); values are clipped with a `RuntimeWarning`.
- `get_hist` on a grayscale array with out-of-range values now warns (`RuntimeWarning`)
  instead of clipping silently.
- `array_to_image` on an empty array raises a clear `ValueError` ("arr must contain at least
  one pixel") instead of a NumPy reduction error; zero or negative `width`/`height` are
  rejected with a message naming the parameter.
- Float pixel values are rounded to the nearest integer (round half to even) instead of
  truncated (127.9 became 127).
- NaN or infinite values raise `ValueError` instead of being counted in bin 0 with a NumPy
  cast warning. Non-numeric arrays raise `TypeError`.
- Boolean arrays map to 0/255 (the usual mask-to-image convention).
- Clipping warnings now point at the caller's line, not at library internals.
- `array_to_image` no longer passes Pillow's deprecated `mode=` argument to `Image.fromarray`.
- `__version__` now matches the released version (it was stuck at 1.1.1).
- `get_hist` no longer makes an unnecessary copy of array input.

### Documentation

- `get_image_array` documents that 1.x output is always 8-bit (16-bit and float images are
  clipped); native 16-bit/float support is planned for 2.0.

## [1.1.8] - 2026-02-09

- Test fix (out-of-range test handles multiple warnings). No functional changes.

## [1.1.7] - 2026-02-09

- CI/CD improvements and an expanded test suite (edge cases, errors, deprecated wrappers).
  No functional changes.

## [1.1.3]–[1.1.6] - 2026-01-16 to 2026-01-24

- Maintenance releases fixing the CI and PyPI publish workflows. No functional changes.

## [1.1.2] - 2026-01-16

- Added CONTRIBUTING and Code of Conduct, ruff linting in CI, tag-based PyPI publishing and
  README badges. No functional changes.

## [1.1.1] - 2025-08-12

### Changed

- Removed author email from package metadata for privacy.

### Security

- Metadata-only change; no code modifications.

## [1.1.0] - 2025-08-12

### Added

- New PEP-8 function names: `get_image_array`, `get_hist`, `array_to_image`.
- Numpy vectorized histogram implementation (`numpy.bincount`).
- Validation and normalization options (e.g. `as_density` in `get_hist`).
- Automatic grayscale conversion for color inputs.
- Test suite (pytest) covering core behaviors.
- GitHub Actions CI (multi-Python matrix) and build artifact workflow.
- `pyproject.toml` build metadata and development extras.
- `.gitignore` tailored for Python/project artifacts.

### Changed

- Array shape ordering bug fixed: arrays now use `(height, width)` (previously effectively transposed).
- Modernized packaging metadata (keywords, classifiers, dependencies explicitly declared).

### Deprecated

- Legacy camelCase APIs (`getImageArray`, `getHist`, `getImageFromArray`) retained with `DeprecationWarning`. These are slated for removal in a future 2.0.0 release.

### Removed

- Nothing removed; backward compatibility wrappers provided.

### Security

- None.

## [1.0.2] - 2021-02-05

- Historical release (original functionality, camelCase API only).

---

Semantic Versioning is used: MAJOR.MINOR.PATCH.
