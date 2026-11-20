"""Core implementation for the *image_array_and_histogram* package.

Public, preferred (PEP-8) function names:
  - get_image_array
  - get_hist
  - array_to_image

Backward compatible aliases with the historical camelCase API are provided
(getImageArray, getHist, getImageFromArray) and emit a DeprecationWarning when
used.

All arrays follow the conventional NumPy image layout of (height, width).
Earlier versions (<=1.0.x) produced arrays shaped (width, height).
"""

from __future__ import annotations

import warnings
from typing import Sequence, Union

import numpy as np
from PIL import Image

ArrayLike = Union[Sequence[int], np.ndarray]


def _to_uint8(arr: np.ndarray, *, on_out_of_range: str, what: str) -> np.ndarray:
    """Convert an array to uint8 without hidden data corruption.

    on_out_of_range: "clip-warn" -> clip to [0, 255] and emit RuntimeWarning
                     "raise"     -> raise ValueError
    Floats are rounded (np.rint), never truncated. NaN/inf always raise.
    Booleans map to 0 / 255.
    """
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8) * np.uint8(255)
    if arr.size == 0:
        return arr.astype(np.uint8)
    if arr.dtype.kind == "f":
        if not np.isfinite(arr).all():
            raise ValueError(f"{what} contains NaN or infinite values")
        arr = np.rint(arr)
    elif arr.dtype.kind not in "iu":
        raise TypeError(f"{what} must be numeric, got dtype {arr.dtype}")
    lo, hi = arr.min(), arr.max()
    if lo < 0 or hi > 255:
        if on_out_of_range == "raise":
            raise ValueError(f"{what} values must be in [0, 255]; got range [{lo}, {hi}]")
        warnings.warn(
            f"{what} values outside [0, 255] were clipped (range was [{lo}, {hi}])",
            RuntimeWarning,
            stacklevel=3,
        )
        arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def get_image_array(image: Image.Image, ensure_grayscale: bool = True) -> np.ndarray:
    """Return the image as a 2D NumPy uint8 array (height, width).

    Parameters
    ----------
    image : PIL.Image.Image
        Input PIL image.
    ensure_grayscale : bool, default True
        If True and the image is not already in mode 'L', it is converted.

    Notes
    -----
    In 1.x the result is always 8-bit. 16-bit (``I;16``/``I``) and float
    (``F``) images are clipped to [0, 255] (with a ``RuntimeWarning``) when
    ``ensure_grayscale=False``, or collapsed by Pillow's ``convert("L")``
    otherwise. Native 16-bit and float support arrives in 2.0.
    """
    if ensure_grayscale and image.mode != "L":
        image = image.convert("L")
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image; got shape {arr.shape!r}")
    return _to_uint8(arr, on_out_of_range="clip-warn", what="image")


def get_hist(
    image_or_array: Image.Image | ArrayLike, *, as_density: bool = False
) -> list[int] | list[float]:
    """Return the 256-bin histogram for a grayscale image/array.

    Parameters
    ----------
    image_or_array : Image or array-like
        A PIL Image (any mode accepted; converted to 'L'), a 2D array of
        pixel intensities 0..255, or an (H, W, 3/4) RGB(A) array (converted to
        grayscale with Pillow; alpha is dropped).
    as_density : bool, default False
        If True return normalized probabilities summing to 1.0; otherwise counts.

    Notes
    -----
    Float values are rounded to the nearest integer. Values outside [0, 255]
    are clipped with a ``RuntimeWarning``. NaN or infinite values raise
    ``ValueError``.
    """
    if isinstance(image_or_array, Image.Image):
        arr = get_image_array(image_or_array, ensure_grayscale=True)
    else:
        arr = np.asarray(image_or_array)
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):  # RGB(A) passed directly
            arr = _to_uint8(arr, on_out_of_range="clip-warn", what="image_or_array")
            rgb = arr[..., :3] if arr.shape[-1] == 4 else arr
            arr = np.asarray(Image.fromarray(np.ascontiguousarray(rgb)).convert("L"))
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D array of grayscale pixels; got shape {arr.shape!r}")
        arr = _to_uint8(arr, on_out_of_range="clip-warn", what="image_or_array")

    counts = np.bincount(arr.ravel(), minlength=256)
    if not as_density:
        return counts.tolist()
    total = counts.sum()
    if total == 0:
        return [0.0] * 256
    return (counts / total).tolist()


def array_to_image(
    arr: ArrayLike,
    width: int | None = None,
    height: int | None = None,
    *,
    validate_range: bool = True,
) -> Image.Image:
    """Create a grayscale PIL Image from an array.

    Parameters
    ----------
    arr : array-like
        1D sequence of pixels or a 2D ndarray/list. If 1D, width & height must
        be supplied. If 2D, width/height are inferred and must match when
        provided.
    width, height : int, optional
        Dimensions (required for flat 1D input). Ignored for 2D input unless
        provided for validation. Must be positive.
    validate_range : bool, default True
        If True, values outside [0, 255] are clipped with a ``RuntimeWarning``.
        If False, values outside [0, 255] raise ``ValueError``.

    Notes
    -----
    Float values are rounded to the nearest integer (round half to even),
    never truncated. Boolean arrays map ``False -> 0`` and ``True -> 255``.
    NaN or infinite values raise ``ValueError``.
    """
    for name, value in (("width", width), ("height", height)):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value}")

    np_arr = np.asarray(arr)
    if np_arr.ndim == 1:
        if width is None or height is None:
            raise ValueError("width and height are required when passing a 1D array")
        expected = width * height
        if np_arr.size != expected:
            raise ValueError(
                f"Flat array length {np_arr.size} does not equal width*height {expected}"
            )
        np_arr = np_arr.reshape(height, width)  # (H, W)
    elif np_arr.ndim == 2:
        h, w = np_arr.shape
        if width is not None and width != w:
            raise ValueError(f"Provided width {width} does not match array width {w}")
        if height is not None and height != h:
            raise ValueError(f"Provided height {height} does not match array height {h}")
    else:
        raise ValueError("Only 1D or 2D arrays supported for grayscale images")

    if np_arr.size == 0:
        raise ValueError("arr must contain at least one pixel")

    np_arr = _to_uint8(
        np_arr,
        on_out_of_range="clip-warn" if validate_range else "raise",
        what="arr",
    )
    return Image.fromarray(np.ascontiguousarray(np_arr))


# ---------------------------------------------------------------------------
# Backward compatibility wrappers (deprecated camelCase names)
# ---------------------------------------------------------------------------


def getImageArray(image_object):
    """Deprecated alias of :func:`get_image_array`."""
    warnings.warn(
        "getImageArray is deprecated; use get_image_array", DeprecationWarning, stacklevel=2
    )
    return get_image_array(image_object)


def getHist(image_arr):
    """Deprecated alias of :func:`get_hist`."""
    warnings.warn("getHist is deprecated; use get_hist", DeprecationWarning, stacklevel=2)
    return get_hist(image_arr)


def getImageFromArray(arr, width, height):
    """Deprecated alias of :func:`array_to_image`."""
    warnings.warn(
        "getImageFromArray is deprecated; use array_to_image "
        "(note: shape semantics changed to (height,width))",
        DeprecationWarning,
        stacklevel=2,
    )
    return array_to_image(arr, width=width, height=height)
