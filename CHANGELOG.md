# Changelog

All notable changes to this project will be documented in this file.

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
