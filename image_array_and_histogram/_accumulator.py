"""Streaming histogram accumulation over many images."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ._convert import _check_channels, _to_array
from ._histogram import (
    ValueRange,
    _check_bins,
    _check_range,
    _histogram_with_edges,
    _is_dtype_range,
    default_range,
)
from ._stats import DEFAULT_PERCENTILES, HistStats, stats_for
from ._types import Channels, ImageInput


class HistAccumulator:
    """Accumulate one histogram over many images, one image at a time.

    Parameters
    ----------
    channels : {"gray", "rgb"}, default "gray"
        Histogram the luma or each RGB channel.
    bins : int, default 256
        Number of bins.
    value_range : (float, float), optional
        Range covered by the bins, fixed for all images. If omitted it is
        taken from the first image's dtype (uint8 -> (0, 256), uint16 ->
        (0, 65536)). Float and other integer images need an explicit range,
        because their default range depends on each image's data.

    Examples
    --------
    >>> acc = HistAccumulator()
    >>> for img in images:  # doctest: +SKIP
    ...     acc.update(img)
    >>> acc.stats()  # doctest: +SKIP
    """

    __slots__ = ("_bins", "_channels", "_counts", "_explicit_range", "_n_images", "_range")

    def __init__(
        self,
        *,
        channels: Channels = "gray",
        bins: int = 256,
        value_range: tuple[float, float] | None = None,
    ) -> None:
        _check_channels(channels)
        self._channels: Channels = channels
        self._bins = _check_bins(bins)
        self._explicit_range = value_range is not None
        self._range: ValueRange | None = None if value_range is None else _check_range(value_range)
        shape = (self._bins,) if channels == "gray" else (3, self._bins)
        self._counts = np.zeros(shape, dtype=np.int64)
        self._n_images = 0

    def __repr__(self) -> str:
        return (
            f"HistAccumulator(channels={self._channels!r}, bins={self._bins}, "
            f"value_range={self._range}, n_images={self._n_images})"
        )

    def _range_for(self, arr: NDArray[np.generic]) -> ValueRange:
        if self._explicit_range:
            assert self._range is not None
            return self._range
        if not _is_dtype_range(arr.dtype):
            raise ValueError(
                f"{arr.dtype} images need an explicit value_range: their default range "
                "depends on each image's data, so bins would not line up across images"
            )
        rng = default_range(arr)
        if self._range is not None and rng != self._range:
            raise ValueError(
                f"image dtype {arr.dtype} implies value_range {rng}, but earlier images used "
                f"{self._range}; pass value_range explicitly to mix bit depths"
            )
        return rng

    def update(self, image: ImageInput) -> None:
        """Add one image's histogram to the running total.

        Parameters
        ----------
        image : PIL.Image.Image or array-like
            The image (see :func:`get_hist`).

        Raises
        ------
        ValueError
            For invalid input, or when the image's dtype doesn't fit the
            accumulator's range (see the class docstring).
        """
        arr = _to_array(image, self._channels)
        rng = self._range_for(arr)
        counts, _ = _histogram_with_edges(arr, self._bins, rng)
        self._range = rng
        self._counts += counts
        self._n_images += 1

    def merge(self, other: HistAccumulator) -> None:
        """Add another accumulator's counts (for example from a worker process).

        Parameters
        ----------
        other : HistAccumulator
            An accumulator with the same ``channels``, ``bins`` and range.

        Raises
        ------
        TypeError
            If ``other`` is not a ``HistAccumulator``.
        ValueError
            If the configurations differ.
        """
        if not isinstance(other, HistAccumulator):
            raise TypeError(f"can only merge a HistAccumulator, got {type(other).__name__}")
        if (other._channels, other._bins) != (self._channels, self._bins):
            raise ValueError(
                "cannot merge accumulators with different channels or bins: "
                f"({self._channels}, {self._bins}) vs ({other._channels}, {other._bins})"
            )
        if other._n_images == 0:
            return
        if self._range is not None and other._range != self._range:
            raise ValueError(
                f"cannot merge accumulators with different ranges: {self._range} vs {other._range}"
            )
        self._range = other._range
        self._counts += other._counts
        self._n_images += other._n_images

    @property
    def counts(self) -> NDArray[np.int64]:
        """Accumulated counts, ``(bins,)`` or ``(3, bins)``, as a read-only view."""
        view = self._counts.view()
        view.flags.writeable = False
        return view

    @property
    def edges(self) -> NDArray[np.float64]:
        """Bin edges, shape ``(bins + 1,)``.

        Raises
        ------
        ValueError
            If no range is known yet (no ``value_range`` and no images added).
        """
        if self._range is None:
            raise ValueError("edges are unknown until an image is added or value_range is set")
        return np.linspace(*self._range, self._bins + 1)

    @property
    def n_images(self) -> int:
        """Number of images added (including merged ones)."""
        return self._n_images

    @property
    def channels(self) -> Channels:
        """The configured channels."""
        return self._channels

    @property
    def bins(self) -> int:
        """The configured number of bins."""
        return self._bins

    @property
    def value_range(self) -> tuple[float, float] | None:
        """The bin range, or ``None`` until it is known."""
        return self._range

    def stats(
        self, percentiles: Sequence[float] = DEFAULT_PERCENTILES
    ) -> HistStats | dict[str, HistStats]:
        """Compute statistics of the accumulated histogram.

        Parameters
        ----------
        percentiles : sequence of float, default (1, 5, 25, 50, 75, 95, 99)
            Percentiles to report.

        Returns
        -------
        HistStats or dict of str to HistStats
            See :func:`hist_stats`.

        Raises
        ------
        ValueError
            If nothing has been accumulated yet.
        """
        if self._n_images == 0 or self._range is None:
            raise ValueError("cannot compute statistics of an empty histogram")
        return stats_for(self._counts, self.edges, percentiles)
