"""Shared type aliases."""

from __future__ import annotations

from typing import Literal, TypeAlias

from numpy.typing import ArrayLike
from PIL import Image

Channels: TypeAlias = Literal["gray", "rgb"]
"""Which channels to analyse: ``"gray"`` (luma) or ``"rgb"`` (one row per channel)."""

ImageInput: TypeAlias = Image.Image | ArrayLike
"""A PIL image or anything :func:`numpy.asarray` accepts."""

CHANNEL_NAMES: tuple[str, str, str] = ("r", "g", "b")
