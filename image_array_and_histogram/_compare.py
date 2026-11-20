"""Histogram comparison metrics."""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

CompareMethod: TypeAlias = Literal["intersection", "chi2", "bhattacharyya", "correlation", "emd"]

#: For each method: (range, value meaning "identical").
METHOD_INFO: dict[str, tuple[str, float]] = {
    "intersection": ("[0, 1]", 1.0),
    "chi2": ("[0, 1]", 0.0),
    "bhattacharyya": ("[0, 1]", 0.0),
    "correlation": ("[-1, 1]", 1.0),
    "emd": ("[0, bins - 1]", 0.0),
}


def _as_hist(h: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(h)
    if arr.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be numeric, got dtype {arr.dtype}")
    arr = arr.astype(np.float64, copy=False)
    if arr.ndim not in (1, 2) or arr.shape[-1] == 0:
        raise ValueError(f"{name} must be a non-empty 1-D or 2-D histogram, got shape {arr.shape}")
    if not np.isfinite(arr).all() or np.any(arr < 0):
        raise ValueError(f"{name} must contain finite, non-negative counts")
    return arr


def _normalise(h: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    totals = h.sum(axis=-1, keepdims=True)
    if np.any(totals == 0):
        raise ValueError(f"{name} contains an all-zero histogram")
    return h / totals


def _correlation(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    da = a - a.mean(axis=-1, keepdims=True)
    db = b - b.mean(axis=-1, keepdims=True)
    num = (da * db).sum(axis=-1)
    den = np.sqrt((da * da).sum(axis=-1) * (db * db).sum(axis=-1))
    constant = den == 0
    equal = np.all(a == b, axis=-1)
    out = np.where(constant, np.where(equal, 1.0, 0.0), num / np.where(constant, 1.0, den))
    return np.clip(out, -1.0, 1.0)


def compare_hist(h1: ArrayLike, h2: ArrayLike, *, method: CompareMethod = "intersection") -> float:
    """Compare two histograms.

    Parameters
    ----------
    h1, h2 : array-like
        Histograms of the same shape: ``(bins,)``, or ``(channels, bins)`` such
        as the ``(3, bins)`` output of ``get_hist(..., channels="rgb")``.
    method : str, default "intersection"
        The metric. Except for ``"correlation"``, each histogram is first
        normalised to sum to 1, so images of different sizes compare fairly.

        ===============  ===============  ===========  ==================================
        method           range            identical    definition
        ===============  ===============  ===========  ==================================
        intersection     [0, 1]           1            sum(min(p, q))
        chi2             [0, 1]           0            0.5 * sum((p - q)^2 / (p + q))
        bhattacharyya    [0, 1]           0            sqrt(1 - sum(sqrt(p * q)))
        correlation      [-1, 1]          1            Pearson r of the raw counts
        emd              [0, bins - 1]    0            sum(abs(cumsum(p) - cumsum(q)))
        ===============  ===============  ===========  ==================================

        ``emd`` is the 1-D Earth Mover's (Wasserstein-1) distance in bin units.
        For ``correlation``, if either histogram is constant the result is 1.0
        when they are equal and 0.0 otherwise.

    Returns
    -------
    float
        The score. For 2-D input the per-row scores are averaged.

    Raises
    ------
    ValueError
        For mismatched or invalid shapes, negative or non-finite counts, an
        all-zero histogram (except with ``correlation``), or an unknown method.
    TypeError
        For non-numeric input.
    """
    if method not in METHOD_INFO:
        raise ValueError(f"method must be one of {sorted(METHOD_INFO)}, got {method!r}")
    a, b = _as_hist(h1, "h1"), _as_hist(h2, "h2")
    if a.shape != b.shape:
        raise ValueError(f"histogram shapes differ: {a.shape} vs {b.shape}")

    if method == "correlation":
        scores = _correlation(a, b)
    else:
        p, q = _normalise(a, "h1"), _normalise(b, "h2")
        if method == "intersection":
            scores = np.minimum(p, q).sum(axis=-1)
        elif method == "chi2":
            s = p + q
            d = (p - q) ** 2
            scores = 0.5 * np.divide(d, s, out=np.zeros_like(s), where=s > 0).sum(axis=-1)
        elif method == "bhattacharyya":
            bc = np.sqrt(p * q).sum(axis=-1)
            scores = np.sqrt(np.maximum(0.0, 1.0 - bc))
        else:  # emd
            scores = np.abs(np.cumsum(p, axis=-1) - np.cumsum(q, axis=-1)).sum(axis=-1)
        if method != "emd":
            scores = np.clip(scores, 0.0, 1.0)
    return float(np.mean(scores))
