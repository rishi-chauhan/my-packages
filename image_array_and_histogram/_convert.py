"""Input normalisation: PIL images and array-likes to NumPy arrays, and back."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image

from ._types import Channels, ImageInput

# PIL modes whose pixel data is returned as-is (16/32-bit integer or float, single band).
_NATIVE_MODES = frozenset({"I;16", "I;16B", "I;16L", "I;16N", "I", "F"})

# ITU-R 601-2 luma weights, the same ones Pillow uses for convert("L").
_LUMA = (0.299, 0.587, 0.114)


def _check_channels(channels: str) -> None:
    if channels not in ("gray", "rgb"):
        raise ValueError(f"channels must be 'gray' or 'rgb', got {channels!r}")


def _native_byteorder(arr: NDArray[Any]) -> NDArray[Any]:
    """Return ``arr`` with native byte order (``I;16B`` images give big-endian data)."""
    if arr.dtype.byteorder not in ("=", "|"):
        return arr.astype(arr.dtype.newbyteorder("="))
    return arr


def _check_values(arr: NDArray[Any], what: str) -> NDArray[Any]:
    """Validate dtype and finiteness; map bool to uint8 0/255."""
    kind = arr.dtype.kind
    if kind == "b":
        return arr.astype(np.uint8) * np.uint8(255)
    if kind == "f":
        if arr.size and not np.isfinite(arr).all():
            raise ValueError(f"{what} contains NaN or infinite values")
        return arr
    if kind in "iu":
        return arr
    raise TypeError(f"{what} must be a numeric array, got dtype {arr.dtype}")


def _rgb_to_gray(rgb: NDArray[Any]) -> NDArray[Any]:
    """Convert an (H, W, 3) array to (H, W) luma, keeping the input dtype."""
    if rgb.size == 0:
        return np.empty(rgb.shape[:2], dtype=rgb.dtype)
    if rgb.dtype == np.uint8:
        # Pillow's C routine: identical results to PIL-image input, and fast.
        return np.asarray(Image.fromarray(np.ascontiguousarray(rgb)).convert("L"))
    # float32 is enough for 8/16-bit data; wider types need float64 to stay exact.
    work = np.float32 if rgb.dtype.itemsize <= 2 or rgb.dtype == np.float32 else np.float64
    f: NDArray[np.floating[Any]] = rgb.astype(work, copy=False)
    gray: NDArray[np.floating[Any]] = f[..., 0] * _LUMA[0] + f[..., 1] * _LUMA[1]
    gray += f[..., 2] * _LUMA[2]
    if rgb.dtype.kind == "f":
        return gray.astype(rgb.dtype, copy=False)
    info = np.iinfo(rgb.dtype)
    out: NDArray[Any] = np.clip(np.rint(gray), info.min, info.max).astype(rgb.dtype)
    return out


def _pil_to_array(image: Image.Image, channels: Channels) -> NDArray[Any]:
    if image.mode in _NATIVE_MODES:
        if channels == "rgb":
            raise ValueError(f"channels='rgb' requires a colour image; got mode {image.mode!r}")
        arr = _native_byteorder(np.asarray(image))
        # Some decoders (and older Pillow versions) deliver 16-bit data as 32-bit mode "I";
        # treat data that fits in 16 bits as uint16 so results don't depend on the source.
        if image.mode == "I" and arr.size and arr.min() >= 0 and arr.max() <= 65535:
            return arr.astype(np.uint16)
        return arr
    return np.asarray(image.convert("L" if channels == "gray" else "RGB"))


def _to_array(image: ImageInput, channels: Channels) -> NDArray[Any]:
    """Return (H, W) for ``"gray"`` or (H, W, 3) for ``"rgb"``.

    The native dtype is preserved (uint8, uint16, other integers or float);
    data is never silently down-converted. Booleans become uint8 0/255.
    """
    _check_channels(channels)
    if isinstance(image, Image.Image):
        return _pil_to_array(image, channels)

    arr = _check_values(_native_byteorder(np.asarray(image)), "image")
    if arr.ndim == 2:
        if channels == "rgb":
            raise ValueError(
                f"channels='rgb' requires an (H, W, 3) or (H, W, 4) array; got shape {arr.shape}"
            )
        return arr
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3]
        return rgb if channels == "rgb" else _rgb_to_gray(rgb)
    raise ValueError(f"image must have shape (H, W), (H, W, 3) or (H, W, 4); got shape {arr.shape}")


def _to_uint8(arr: NDArray[Any], *, clip: bool, what: str) -> NDArray[np.uint8]:
    """Convert to uint8: round floats, clip with a warning (or raise) when out of range.

    Expects a non-empty array that passed :func:`_check_values` (numeric, finite, no bool).
    """
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype.kind == "f":
        arr = np.rint(arr)
    lo, hi = arr.min(), arr.max()
    if lo < 0 or hi > 255:
        if not clip:
            raise ValueError(f"{what} values must be in [0, 255]; got range [{lo}, {hi}]")
        warnings.warn(
            f"{what} values outside [0, 255] were clipped (range was [{lo}, {hi}])",
            RuntimeWarning,
            stacklevel=3,
        )
        arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def get_image_array(
    image: Image.Image, *, mode: Literal["gray", "rgb", "keep"] = "gray"
) -> NDArray[Any]:
    """Return a PIL image as a NumPy array.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image.
    mode : {"gray", "rgb", "keep"}, default "gray"
        ``"gray"`` returns (H, W) luma; ``"rgb"`` returns (H, W, 3);
        ``"keep"`` returns ``np.asarray(image)`` unchanged.

    Returns
    -------
    numpy.ndarray
        The pixel data. The native bit depth is kept: 8-bit images give uint8,
        16-bit images (``I;16``) give uint16, ``I`` gives int32 (uint16 when all
        values fit in 16 bits, since some decoders deliver 16-bit data as ``I``) and
        ``F`` gives float32.

    Raises
    ------
    TypeError
        If ``image`` is not a PIL image.
    ValueError
        If ``mode`` is invalid, or ``mode="rgb"`` is used on a 16/32-bit or
        float single-band image.
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    if mode == "keep":
        return np.asarray(image)
    if mode not in ("gray", "rgb"):
        raise ValueError(f"mode must be 'gray', 'rgb' or 'keep', got {mode!r}")
    return _pil_to_array(image, mode)


def array_to_image(
    arr: ArrayLike,
    width: int | None = None,
    height: int | None = None,
    *,
    clip: bool = True,
) -> Image.Image:
    """Create a PIL image from an array.

    Parameters
    ----------
    arr : array-like
        A 1-D sequence (``width`` and ``height`` required), an (H, W) array
        (mode ``L``, or ``I;16`` for uint16), or an (H, W, 3) / (H, W, 4) array
        (mode ``RGB`` / ``RGBA``).
    width, height : int, optional
        Image size. Required for 1-D input; for other shapes they are checked
        against the array when given. Must be positive.
    clip : bool, default True
        How to handle values outside [0, 255] when converting to 8 bits:
        ``True`` clips them and emits a ``RuntimeWarning``; ``False`` raises
        ``ValueError``.

    Returns
    -------
    PIL.Image.Image
        The image.

    Raises
    ------
    ValueError
        For invalid shapes or sizes, empty input, NaN/inf values, or
        out-of-range values with ``clip=False``.
    TypeError
        For non-numeric arrays.

    Notes
    -----
    uint8 data is used as-is and 2-D uint16 data becomes a lossless ``I;16``
    image. Everything else is converted to uint8: floats are rounded to the
    nearest integer (round half to even), never truncated, and booleans map
    ``False -> 0``, ``True -> 255``.
    """
    for name, value in (("width", width), ("height", height)):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value}")

    np_arr = _check_values(_native_byteorder(np.asarray(arr)), "arr")
    if np_arr.ndim == 1:
        if width is None or height is None:
            raise ValueError("width and height are required when passing a 1D array")
        if np_arr.size != width * height:
            raise ValueError(
                f"Flat array length {np_arr.size} does not equal width*height {width * height}"
            )
        np_arr = np_arr.reshape(height, width)
    elif np_arr.ndim == 2 or (np_arr.ndim == 3 and np_arr.shape[-1] in (3, 4)):
        h, w = np_arr.shape[:2]
        if width is not None and width != w:
            raise ValueError(f"Provided width {width} does not match array width {w}")
        if height is not None and height != h:
            raise ValueError(f"Provided height {height} does not match array height {h}")
    else:
        raise ValueError(
            f"arr must have shape (N,), (H, W), (H, W, 3) or (H, W, 4); got {np_arr.shape}"
        )

    if np_arr.size == 0:
        raise ValueError("arr must contain at least one pixel")
    if not (np_arr.ndim == 2 and np_arr.dtype == np.uint16):
        np_arr = _to_uint8(np_arr, clip=clip, what="arr")
    return Image.fromarray(np.ascontiguousarray(np_arr))
