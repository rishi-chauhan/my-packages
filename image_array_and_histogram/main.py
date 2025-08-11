"""Core implementation for the *image_array_and_histogram* package.

Public, preferred (PEP-8) function names:
  - get_image_array
  - get_hist
  - array_to_image

Backward compatible aliases with the historical camelCase API are provided
in ``__init__.py`` (getImageArray, getHist, getImageFromArray) and emit a
DeprecationWarning when used.

All arrays now follow the conventional NumPy image layout of (height, width).
Earlier versions (<=1.0.x) produced arrays shaped (width, height). This was
confusing and is considered a bug fix; the minor version has been bumped.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Union, Optional
import warnings

import numpy as np
from PIL import Image

ArrayLike = Union[Sequence[int], np.ndarray]

def _ensure_uint8(arr: np.ndarray) -> np.ndarray:
    """Coerce an array to dtype uint8 (0-255) safely."""
    if arr.dtype != np.uint8:
        # Clip to range then cast
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def get_image_array(image: Image.Image, ensure_grayscale: bool = True) -> np.ndarray:
    """Return the image as a 2D NumPy uint8 array (height, width).

    Parameters
    ----------
    image : PIL.Image.Image
        Input PIL image.
    ensure_grayscale : bool, default True
        If True and the image is not already in mode 'L', it is converted.
    """
    if ensure_grayscale and image.mode != "L":
        image = image.convert("L")
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D grayscale image; got shape %r" % (arr.shape,))
    return _ensure_uint8(arr)

def get_hist(image_or_array: Union[Image.Image, ArrayLike], *, as_density: bool = False) -> list[int] | list[float]:
    """Return the 256-bin histogram for a grayscale image/array.

    Parameters
    ----------
    image_or_array : Image or array-like
        A grayscale PIL Image (any mode accepted; converted to 'L') or a 2D
        NumPy/list-like of pixel intensities 0..255.
    as_density : bool, default False
        If True return normalized probabilities summing to 1.0; otherwise counts.
    """
    if isinstance(image_or_array, Image.Image):
        arr = get_image_array(image_or_array, ensure_grayscale=True)
    else:
        arr = np.array(image_or_array)
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):  # RGB(A) passed directly
            # Convert through PIL to reuse conversion logic
            pil = Image.fromarray(arr.astype(np.uint8))
            arr = get_image_array(pil, ensure_grayscale=True)
        if arr.ndim != 2:
            raise ValueError("Expected a 2D array of grayscale pixels; got shape %r" % (arr.shape,))
        arr = _ensure_uint8(arr)

    counts = np.bincount(arr.ravel(), minlength=256)[:256]
    if not as_density:
        return counts.tolist()
    total = counts.sum()
    if total == 0:
        return [0.0] * 256
    return (counts / total).astype(float).tolist()

def array_to_image(arr: ArrayLike, width: Optional[int] = None, height: Optional[int] = None, *, validate_range: bool = True) -> Image.Image:
    """Create a grayscale PIL Image from an array.

    Parameters
    ----------
    arr : array-like
        1D sequence of pixels or a 2D ndarray/list. If 1D, width & height must
        be supplied. If 2D, width/height are inferred and must match when
        provided.
    width, height : int, optional
        Dimensions (required for flat 1D input). Ignored for 2D input unless
        provided for validation.
    validate_range : bool, default True
        If True clamp values to [0,255]; otherwise values outside range raise.
    """
    np_arr = np.array(arr)
    if np_arr.ndim == 1:
        if width is None or height is None:
            raise ValueError("width and height are required when passing a 1D array")
        expected = width * height
        if np_arr.size != expected:
            raise ValueError(f"Flat array length {np_arr.size} does not equal width*height {expected}")
        np_arr = np_arr.reshape(height, width)  # (H, W)
    elif np_arr.ndim == 2:
        h, w = np_arr.shape
        if width is not None and width != w:
            raise ValueError(f"Provided width {width} does not match array width {w}")
        if height is not None and height != h:
            raise ValueError(f"Provided height {height} does not match array height {h}")
    else:
        raise ValueError("Only 1D or 2D arrays supported for grayscale images")

    if validate_range:
        if np_arr.min() < 0 or np_arr.max() > 255:
            warnings.warn("Pixel values clipped to [0,255]", RuntimeWarning)
            np_arr = np.clip(np_arr, 0, 255)
    np_arr = _ensure_uint8(np_arr)
    return Image.fromarray(np_arr, mode="L")

# ---------------------------------------------------------------------------
# Backward compatibility wrappers (deprecated camelCase names)
# ---------------------------------------------------------------------------

def getImageArray(image_object):  # pragma: no cover - thin wrapper
    warnings.warn("getImageArray is deprecated; use get_image_array", DeprecationWarning, stacklevel=2)
    return get_image_array(image_object)

def getHist(image_arr):  # pragma: no cover
    warnings.warn("getHist is deprecated; use get_hist", DeprecationWarning, stacklevel=2)
    return get_hist(image_arr)

def getImageFromArray(arr, width, height):  # pragma: no cover
    warnings.warn("getImageFromArray is deprecated; use array_to_image (note: shape semantics changed to (height,width))", DeprecationWarning, stacklevel=2)
    return array_to_image(arr, width=width, height=height)

