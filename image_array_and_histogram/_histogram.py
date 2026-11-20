"""Histogram core: bin selection, fast counting paths and :func:`get_hist`."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._convert import _to_array
from ._types import Channels, ImageInput

ValueRange = tuple[float, float]


def _check_bins(bins: int) -> int:
    if isinstance(bins, bool):
        raise ValueError(f"bins must be a positive integer, got {bins!r}")
    try:
        n = operator.index(bins)
    except TypeError:
        raise ValueError(f"bins must be a positive integer, got {bins!r}") from None
    if n <= 0:
        raise ValueError(f"bins must be a positive integer, got {bins!r}")
    return n


def _check_range(value_range: ValueRange) -> ValueRange:
    try:
        lo, hi = (float(v) for v in value_range)
    except (TypeError, ValueError):
        raise ValueError(f"value_range must be a (low, high) pair, got {value_range!r}") from None
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError(f"value_range must be finite with low < high, got {value_range!r}")
    return lo, hi


def _is_dtype_range(dtype: np.dtype[Any]) -> bool:
    """Whether the default range of ``dtype`` is fixed (not data-dependent)."""
    return dtype in (np.uint8, np.uint16)


def default_range(arr: NDArray[Any]) -> ValueRange:
    """Return the default histogram range for ``arr``.

    uint8 -> (0, 256); uint16 -> (0, 65536); other integers -> (min, max + 1);
    floats -> (0, 1) if all data lies in [0, 1], otherwise (min, max).
    """
    if arr.dtype == np.uint8:
        return 0.0, 256.0
    if arr.dtype == np.uint16:
        return 0.0, 65536.0
    if arr.size == 0:
        return 0.0, 1.0
    lo, hi = float(arr.min()), float(arr.max())
    if arr.dtype.kind in "iu":
        return lo, hi + 1.0
    if lo >= 0.0 and hi <= 1.0:
        return 0.0, 1.0
    if lo == hi:  # same widening rule as np.histogram
        return lo - 0.5, hi + 0.5
    return lo, hi


def _counts_1d(arr: NDArray[Any], bins: int, lo: float, hi: float) -> NDArray[np.int64]:
    """Count a 1-D array into ``bins`` equal bins over ``[lo, hi]``.

    Matches ``np.histogram(arr, bins, (lo, hi))[0]`` exactly: bins are half-open
    except the last, which includes ``hi``; values outside the range are ignored.
    """
    lo, hi = float(lo), float(hi)
    # 1. The common case: 8-bit data, 256 bins.
    if arr.dtype == np.uint8 and lo == 0 and hi == 256 and bins == 256:
        return np.bincount(arr, minlength=256).astype(np.int64, copy=False)

    # 2. Unsigned integers with integer bin edges that each cover whole values.
    if arr.dtype in (np.uint8, np.uint16) and lo.is_integer() and hi.is_integer():
        ilo, ihi = int(lo), int(hi)
        span = ihi - ilo
        if ilo >= 0 and ihi <= np.iinfo(arr.dtype).max + 1 and span % bins == 0:
            if ilo == 0 and (arr.size == 0 or int(arr.max()) < ihi):
                c = np.bincount(arr, minlength=ihi)
            else:
                sel = arr[(arr >= ilo) & (arr <= ihi)]
                c = np.bincount(sel - arr.dtype.type(ilo), minlength=span + 1)
                c[span - 1] += c[span]  # values == hi fall in the last bin
                c = c[:span]
            summed: NDArray[np.int64] = c.reshape(bins, -1).sum(axis=1, dtype=np.int64)
            return summed

    # 3. Everything else.
    hist: NDArray[np.intp] = np.histogram(arr, bins=bins, range=(lo, hi))[0]
    return hist.astype(np.int64, copy=False)


def _histogram_with_edges(
    arr: NDArray[Any], bins: int, value_range: ValueRange
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Histogram an already-normalised (H, W) or (H, W, 3) array.

    Returns counts of shape ``(bins,)`` or ``(3, bins)`` and ``bins + 1`` edges.
    """
    lo, hi = value_range
    if arr.ndim == 2:
        counts = _counts_1d(arr.ravel(), bins, lo, hi)
    else:  # one channel at a time keeps memory low
        counts = np.stack([_counts_1d(arr[..., i].ravel(), bins, lo, hi) for i in range(3)])
    return counts, np.linspace(lo, hi, bins + 1)


def _density(counts: NDArray[np.int64]) -> NDArray[np.float64]:
    totals = counts.sum(axis=-1, keepdims=True)
    out = np.zeros(counts.shape, dtype=np.float64)
    np.divide(counts, totals, out=out, where=totals > 0)
    return out


def get_hist(
    image: ImageInput,
    *,
    channels: Channels = "gray",
    bins: int = 256,
    value_range: tuple[float, float] | None = None,
    density: bool = False,
) -> NDArray[np.int64] | NDArray[np.float64]:
    """Compute the histogram of an image.

    Parameters
    ----------
    image : PIL.Image.Image or array-like
        A PIL image, an (H, W) array, or an (H, W, 3/4) RGB(A) array (alpha is
        ignored). 8-bit, 16-bit, other integer and float data are supported.
    channels : {"gray", "rgb"}, default "gray"
        ``"gray"`` histograms the luma; ``"rgb"`` histograms each channel.
    bins : int, default 256
        Number of equal-width bins.
    value_range : (float, float), optional
        The ``(low, high)`` range covered by the bins. Defaults depend on the
        dtype: uint8 -> (0, 256), uint16 -> (0, 65536), other integers ->
        (min, max + 1) of the data, floats -> (0, 1) when all data is in
        [0, 1], otherwise (min, max). Values outside the range are ignored,
        as in :func:`numpy.histogram`.
    density : bool, default False
        Return each histogram divided by its total (float64, sums to 1). An
        empty histogram gives all zeros.

    Returns
    -------
    numpy.ndarray
        int64 counts (float64 when ``density=True``) of shape ``(bins,)`` for
        ``"gray"`` or ``(3, bins)`` for ``"rgb"`` (rows R, G, B). Bin edges are
        ``np.linspace(low, high, bins + 1)``; bins are half-open except the
        last, which includes ``high``.

    Raises
    ------
    ValueError
        For invalid shapes, ``bins``, ``value_range`` or ``channels``, NaN/inf
        values, or ``channels="rgb"`` on single-band input.
    TypeError
        For non-numeric arrays.
    """
    n_bins = _check_bins(bins)
    arr = _to_array(image, channels)
    rng = default_range(arr) if value_range is None else _check_range(value_range)
    counts, _ = _histogram_with_edges(arr, n_bins, rng)
    return _density(counts) if density else counts
