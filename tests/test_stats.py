import json

import numpy as np
import pytest

from image_array_and_histogram import HistStats, get_hist, hist_stats, stats_from_counts
from image_array_and_histogram._stats import DEFAULT_PERCENTILES


def test_uint8_stats_match_numpy(rng):
    arr = rng.integers(0, 256, size=(200, 300), dtype=np.uint8)
    s = hist_stats(arr)
    assert isinstance(s, HistStats)
    assert s.count == arr.size
    assert s.mean == pytest.approx(arr.mean(), abs=1e-9)
    assert s.std == pytest.approx(arr.std(), abs=1e-9)
    assert s.min == arr.min()
    assert s.max == arr.max()
    for p in DEFAULT_PERCENTILES:
        assert s.percentiles[p] == np.percentile(arr, p, method="inverted_cdf")
    assert s.median == np.percentile(arr, 50, method="inverted_cdf")


@pytest.mark.parametrize("seed", range(5))
def test_percentiles_match_inverted_cdf_on_skewed_data(seed):
    r = np.random.default_rng(seed)
    arr = np.clip(r.normal(60, 25, size=(57, 31)), 0, 255).astype(np.uint8)
    qs = [0, 0.5, 1, 10, 33.3, 50, 66.6, 90, 99.9, 100]
    s = hist_stats(arr, percentiles=qs)
    for q in qs:
        assert s.percentiles[q] == np.percentile(arr, q, method="inverted_cdf"), q


def test_percentile_zero_is_minimum_even_with_leading_empty_bins():
    arr = np.array([[10, 20, 30]], dtype=np.uint8)
    s = hist_stats(arr, percentiles=[0, 100])
    assert s.percentiles == {0.0: 10.0, 100.0: 30.0}
    assert (s.min, s.max) == (10.0, 30.0)


def test_entropy_and_clipping():
    arr = np.array([[0, 0, 255, 255]], dtype=np.uint8)
    s = hist_stats(arr)
    assert s.entropy == pytest.approx(1.0)
    assert s.clipped_low == 0.5
    assert s.clipped_high == 0.5
    single = hist_stats(np.full((3, 3), 7, np.uint8))
    assert single.entropy == 0.0
    assert single.std == 0.0


def test_rgb_returns_dict(rgb_u8):
    s = hist_stats(rgb_u8, channels="rgb")
    assert set(s) == {"r", "g", "b"}
    assert s["g"].mean == pytest.approx(rgb_u8[..., 1].mean())


def test_bin_centres_used_when_width_not_one():
    counts = np.array([1, 1])
    edges = np.array([0.0, 2.0, 4.0])
    s = stats_from_counts(counts, edges)
    assert s.mean == 2.0  # centres 1 and 3
    assert s.min == 0.0  # left edge of first non-empty bin


def test_float_image_stats_are_approximate(gray_f32):
    s = hist_stats(gray_f32, bins=1000)
    assert s.mean == pytest.approx(float(gray_f32.mean()), abs=1e-3)


def test_to_dict_is_json_serialisable():
    s = hist_stats(np.arange(256, dtype=np.uint8).reshape(16, 16), percentiles=[2.5, 50])
    d = s.to_dict()
    json.dumps(d)
    assert list(d) == [
        "count", "mean", "std", "min", "max", "median", "p2.5", "p50",
        "entropy", "clipped_low", "clipped_high",
    ]  # fmt: skip
    assert d["count"] == 256
    assert d["p50"] == s.median


def test_stats_from_get_hist_counts(gray_u8):
    counts = get_hist(gray_u8)
    s = stats_from_counts(counts, np.arange(257))
    assert s == hist_stats(gray_u8)


def test_hist_stats_respects_bins_and_range(gray_u16):
    s = hist_stats(gray_u16, bins=65536)
    assert s.mean == pytest.approx(gray_u16.mean())
    assert hist_stats(gray_u16, bins=16, value_range=(0, 65536)).count == gray_u16.size


# ---------------------------------------------------------------- errors


def test_empty_histogram_raises():
    with pytest.raises(ValueError, match="empty histogram"):
        hist_stats(np.zeros((0, 0), dtype=np.uint8))
    with pytest.raises(ValueError, match="empty histogram"):
        stats_from_counts(np.zeros(4), np.arange(5))


@pytest.mark.parametrize("percentiles", [[-1], [101], [50, 150]])
def test_bad_percentiles(gray_u8, percentiles):
    with pytest.raises(ValueError, match=r"percentiles must be in \[0, 100\]"):
        hist_stats(gray_u8, percentiles=percentiles)
    with pytest.raises(ValueError, match=r"percentiles must be in \[0, 100\]"):
        stats_from_counts(np.ones(3), np.arange(4), percentiles)


@pytest.mark.parametrize(
    ("counts", "edges"),
    [
        (np.ones(3), np.arange(3)),
        (np.ones((2, 3)), np.arange(4)),
        (np.ones(0), np.arange(1)),
    ],
)
def test_bad_shapes(counts, edges):
    with pytest.raises(ValueError, match="counts must be 1-D"):
        stats_from_counts(counts, edges)


@pytest.mark.parametrize("counts", [np.array([1, -1, 2]), np.array(["a", "b", "c"])])
def test_bad_counts(counts):
    with pytest.raises(ValueError, match="non-negative"):
        stats_from_counts(counts, np.arange(4))
