"""Micro-benchmarks for image_array_and_histogram (not run in CI).

Usage: python benchmarks/bench_hist.py
"""

from __future__ import annotations

import timeit
from collections.abc import Callable

import numpy as np
from PIL import Image

from image_array_and_histogram import get_hist, hist_stats, stats_from_counts

H, W = 3000, 4000  # 12 MP
REPEAT = 7


def best(fn: Callable[[], object], number: int = 3) -> float:
    """Best per-call time in seconds over REPEAT runs."""
    return min(timeit.repeat(fn, number=number, repeat=REPEAT)) / number


def row(name: str, ours: float, ref: float | None = None, target: str = "") -> None:
    ratio = f"{ours / ref:6.2f}x" if ref else "      "
    ref_s = f"{ref * 1e3:9.2f} ms" if ref else " " * 12
    print(f"{name:<44} {ours * 1e3:9.2f} ms  {ref_s}  {ratio}  {target}")


def main() -> None:
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
    rgb = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    gray16 = rng.integers(0, 65536, size=(H, W), dtype=np.uint16)

    print(f"image {W}x{H} ({H * W / 1e6:.0f} MP), numpy {np.__version__}")
    print(f"{'benchmark':<44} {'ours':>12}  {'reference':>12}  {'ratio':>7}  target")

    row(
        "get_hist uint8 gray vs bincount",
        best(lambda: get_hist(gray)),
        best(lambda: np.bincount(gray.ravel(), minlength=256)),
        "<= 1.2x",
    )
    row(
        "get_hist uint8 RGB->gray vs PIL convert+bincount",
        best(lambda: get_hist(rgb)),
        best(
            lambda: np.bincount(
                np.asarray(Image.fromarray(rgb).convert("L")).ravel(), minlength=256
            )
        ),
        "<= 1.2x",
    )
    row("get_hist uint8 channels='rgb'", best(lambda: get_hist(rgb, channels="rgb")))
    row(
        "get_hist uint16 bins=256 vs np.histogram",
        best(lambda: get_hist(gray16, bins=256)),
        best(lambda: np.histogram(gray16, bins=256, range=(0, 65536))),
        "< 1.0x",
    )
    counts = get_hist(gray)
    edges = np.arange(257, dtype=np.float64)
    t = best(lambda: stats_from_counts(counts, edges), number=1000)
    print(f"{'stats_from_counts (256 bins)':<44} {t * 1e6:9.1f} us")
    row("hist_stats uint8 gray (histogram + stats)", best(lambda: hist_stats(gray)))


if __name__ == "__main__":
    main()
