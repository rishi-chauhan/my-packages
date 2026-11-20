"""Convert images to NumPy arrays, compute grayscale histograms, and build images from arrays.

The public API uses PEP-8 function names. Deprecated camelCase aliases are kept
for backward compatibility and emit a ``DeprecationWarning``.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .main import (
    array_to_image,
    get_hist,
    get_image_array,
    getHist,
    getImageArray,
    getImageFromArray,
)

try:
    __version__ = _version("image_array_and_histogram")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without installation
    __version__ = "0+unknown"

__author__ = "Rishi Raj Singh Chauhan"

__all__ = [
    "__version__",
    "array_to_image",
    "getHist",
    "getImageArray",
    "getImageFromArray",
    "get_hist",
    "get_image_array",
]
