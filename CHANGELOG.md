# Changelog

All notable changes to this project will be documented in this file.

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
