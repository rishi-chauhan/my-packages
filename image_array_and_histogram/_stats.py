"""Histogram statistics, computed in O(bins) from counts and edges."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._convert import _to_array
from ._histogram import _check_bins, _check_range, _histogram_with_edges, default_range
from ._types import CHANNEL_NAMES, Channels, ImageInput

DEFAULT_PERCENTILES: tuple[float, ...] = (1, 5, 25, 50, 75, 95, 99)


@dataclass(frozen=True, slots=True)
class HistStats:
    """Summary statistics of one histogram.

    Attributes
    ----------
    count : int
        Number of pixels.
    mean, std : float
        Mean and population standard deviation.
    min, max : float
        Left edge of the first and last non-empty bin (the exact values for
        integer data with bin width 1).
    median : float
        50th percentile.
    percentiles : dict of float to float
        Requested percentiles (nearest-rank / inverted-CDF method).
    entropy : float
        Shannon entropy of the bin distribution, in bits.
    clipped_low, clipped_high : float
        Fraction of pixels in the first bin (crushed shadows) and in the last
        bin (blown highlights).
    """

    count: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    percentiles: dict[float, float]
    entropy: float
    clipped_low: float
    clipped_high: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict; percentile keys look like ``"p5"``."""
        out: dict[str, Any] = {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "median": self.median,
        }
        out.update({f"p{p:g}": v for p, v in self.percentiles.items()})
        out.update(
            entropy=self.entropy,
            clipped_low=self.clipped_low,
            clipped_high=self.clipped_high,
        )
        return out


def _check_percentiles(percentiles: Sequence[float]) -> NDArray[np.float64]:
    q = np.asarray(percentiles, dtype=np.float64).ravel()
    if not np.all((q >= 0) & (q <= 100)):
        raise ValueError(f"percentiles must be in [0, 100], got {list(percentiles)!r}")
    return q


def stats_from_counts(
    counts: ArrayLike,
    edges: ArrayLike,
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
) -> HistStats:
    """Compute statistics from a 1-D histogram.

    Parameters
    ----------
    counts : array-like
        Non-negative counts, shape ``(bins,)``.
    edges : array-like
        Bin edges, shape ``(bins + 1,)``.
    percentiles : sequence of float, default (1, 5, 25, 50, 75, 95, 99)
        Percentiles to report, each in [0, 100].

    Returns
    -------
    HistStats
        The statistics.

    Raises
    ------
    ValueError
        If the shapes don't match, counts are negative, the histogram is
        empty, or a percentile is outside [0, 100].

    Notes
    -----
    Each bin is represented by its left edge when the bin width is 1 (exact
    for integer data, such as the uint8 / 256-bin default) and by its centre
    otherwise, so results are approximate to bin resolution in that case.
    Percentiles use the nearest-rank (inverted CDF) method.
    """
    c = np.asarray(counts)
    e = np.asarray(edges, dtype=np.float64)
    if c.ndim != 1 or e.shape != (c.size + 1,) or c.size == 0:
        raise ValueError(
            f"counts must be 1-D with len(edges) == len(counts) + 1; "
            f"got counts {c.shape}, edges {e.shape}"
        )
    if c.dtype.kind not in "iuf" or np.any(c < 0):
        raise ValueError("counts must be non-negative numbers")
    q = _check_percentiles(percentiles)
    n = float(c.sum())
    if n == 0:
        raise ValueError("cannot compute statistics of an empty histogram")

    widths = np.diff(e)
    values = e[:-1] if np.all(widths == 1.0) else e[:-1] + widths / 2
    mean = float(np.dot(values, c) / n)
    std = float(np.sqrt(np.dot(c, (values - mean) ** 2) / n))
    nonzero = np.flatnonzero(c)

    cdf = np.cumsum(c, dtype=np.float64)
    idx = np.searchsorted(cdf, np.append(q, 50.0) / 100.0 * n, side="left")
    idx = np.clip(np.maximum(idx, nonzero[0]), 0, c.size - 1)
    pvals = values[idx]

    p = c[nonzero] / n
    return HistStats(
        count=int(n),
        mean=mean,
        std=std,
        min=float(e[nonzero[0]]),
        max=float(e[nonzero[-1]]),
        median=float(pvals[-1]),
        percentiles={float(k): float(v) for k, v in zip(q, pvals[:-1], strict=True)},
        entropy=float(-np.sum(p * np.log2(p)) + 0.0),
        clipped_low=float(c[0] / n),
        clipped_high=float(c[-1] / n),
    )


def stats_for(
    counts: NDArray[np.int64], edges: NDArray[np.float64], percentiles: Sequence[float]
) -> HistStats | dict[str, HistStats]:
    """Return stats for a ``(bins,)`` or ``(3, bins)`` histogram."""
    if counts.ndim == 1:
        return stats_from_counts(counts, edges, percentiles)
    return {
        name: stats_from_counts(row, edges, percentiles)
        for name, row in zip(CHANNEL_NAMES, counts, strict=True)
    }


def hist_stats(
    image: ImageInput,
    *,
    channels: Channels = "gray",
    bins: int = 256,
    value_range: tuple[float, float] | None = None,
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
) -> HistStats | dict[str, HistStats]:
    """Compute exposure statistics of an image from its histogram.

    Parameters
    ----------
    image : PIL.Image.Image or array-like
        The image (see :func:`get_hist`).
    channels : {"gray", "rgb"}, default "gray"
        Analyse the luma or each RGB channel.
    bins : int, default 256
        Number of bins.
    value_range : (float, float), optional
        Histogram range (see :func:`get_hist` for the defaults).
    percentiles : sequence of float, default (1, 5, 25, 50, 75, 95, 99)
        Percentiles to report.

    Returns
    -------
    HistStats or dict of str to HistStats
        One ``HistStats`` for ``"gray"``; ``{"r": ..., "g": ..., "b": ...}``
        for ``"rgb"``.

    Raises
    ------
    ValueError
        For invalid input (see :func:`get_hist`) or an empty image.

    Notes
    -----
    Statistics are exact for integer data with bin width 1 (the uint8 default)
    and approximate to bin resolution otherwise.
    """
    n_bins = _check_bins(bins)
    _check_percentiles(percentiles)
    arr = _to_array(image, channels)
    rng = default_range(arr) if value_range is None else _check_range(value_range)
    counts, edges = _histogram_with_edges(arr, n_bins, rng)
    return stats_for(counts, edges, percentiles)
