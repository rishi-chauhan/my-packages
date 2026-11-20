"""Fast image histograms, exposure statistics and histogram comparison for NumPy/Pillow.

Works with PIL images and NumPy arrays: 8-bit, 16-bit and float data, grayscale
or per RGB channel. See :func:`get_hist`, :func:`hist_stats`,
:func:`compare_hist` and :class:`HistAccumulator`.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from ._accumulator import HistAccumulator
from ._compare import compare_hist
from ._convert import array_to_image, get_image_array
from ._histogram import get_hist
from ._stats import HistStats, hist_stats, stats_from_counts

try:
    __version__ = _version("image_array_and_histogram")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without installation
    __version__ = "0+unknown"

__author__ = "Rishi Raj Singh Chauhan"

__all__ = [
    "HistAccumulator",
    "HistStats",
    "__version__",
    "array_to_image",
    "compare_hist",
    "get_hist",
    "get_image_array",
    "hist_stats",
    "stats_from_counts",
]
