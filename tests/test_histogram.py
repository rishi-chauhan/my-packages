import numpy as np
import pytest
from PIL import Image

from image_array_and_histogram import get_hist
from image_array_and_histogram._histogram import _counts_1d, default_range

BINS = [1, 2, 16, 255, 256, 1000]


def _oracle(arr, bins, value_range):
    return np.histogram(arr, bins=bins, range=value_range)[0]


# ---------------------------------------------------------------- oracle checks


@pytest.mark.parametrize("bins", BINS)
@pytest.mark.parametrize("value_range", [None, (0, 256), (0, 255), (10, 200), (0, 64), (5.5, 99.5)])
def test_uint8_matches_np_histogram(gray_u8, bins, value_range):
    rng = value_range or (0, 256)
    np.testing.assert_array_equal(
        get_hist(gray_u8, bins=bins, value_range=value_range), _oracle(gray_u8, bins, rng)
    )


@pytest.mark.parametrize("bins", BINS)
@pytest.mark.parametrize("value_range", [None, (0, 65536), (0, 65535), (1000, 2024), (0, 256)])
def test_uint16_matches_np_histogram(gray_u16, bins, value_range):
    rng = value_range or (0, 65536)
    np.testing.assert_array_equal(
        get_hist(gray_u16, bins=bins, value_range=value_range), _oracle(gray_u16, bins, rng)
    )


@pytest.mark.parametrize("bins", BINS)
@pytest.mark.parametrize("value_range", [None, (0.0, 1.0), (0.25, 0.75)])
def test_float_matches_np_histogram(gray_f32, bins, value_range):
    rng = value_range or (0.0, 1.0)
    np.testing.assert_array_equal(
        get_hist(gray_f32, bins=bins, value_range=value_range), _oracle(gray_f32, bins, rng)
    )


def test_values_equal_to_high_go_in_last_bin():
    arr = np.array([[0, 10, 10, 200]], dtype=np.uint8)
    np.testing.assert_array_equal(get_hist(arr, bins=2, value_range=(0, 10)), [1, 2])
    np.testing.assert_array_equal(
        get_hist(arr, bins=2, value_range=(0, 10)), _oracle(arr, 2, (0, 10))
    )


def test_fast_paths_agree_with_oracle_on_edge_values():
    arr = np.array([0, 1, 254, 255, 255], dtype=np.uint8)
    for lo, hi, bins in [(0, 256, 256), (0, 256, 128), (0, 255, 255), (1, 255, 2), (254, 256, 2)]:
        np.testing.assert_array_equal(_counts_1d(arr, bins, lo, hi), _oracle(arr, bins, (lo, hi)))


def test_rgb_rows_match_single_channel(rgb_u8):
    hist = get_hist(rgb_u8, channels="rgb")
    assert hist.shape == (3, 256)
    for i in range(3):
        np.testing.assert_array_equal(hist[i], get_hist(rgb_u8[..., i]))


def test_rgb_pil_image(rgb_u8):
    np.testing.assert_array_equal(
        get_hist(Image.fromarray(rgb_u8), channels="rgb"), get_hist(rgb_u8, channels="rgb")
    )


def test_gray_rgb_array_matches_pil_image(rgb_u8):
    np.testing.assert_array_equal(get_hist(rgb_u8), get_hist(Image.fromarray(rgb_u8)))


def test_16_bit_image_is_not_collapsed():
    data = np.array([[1000, 40000]], dtype=np.uint16)
    hist = get_hist(Image.fromarray(data))
    assert hist.sum() == 2
    assert hist[1000 // 256] == 1
    assert hist[40000 // 256] == 1


# ---------------------------------------------------------------- default ranges


@pytest.mark.parametrize(
    ("arr", "expected"),
    [
        (np.array([3], np.uint8), (0, 256)),
        (np.array([3], np.uint16), (0, 65536)),
        (np.array([-5, 7], np.int32), (-5, 8)),
        (np.array([0.2, 0.9]), (0.0, 1.0)),
        (np.array([-2.0, 3.0]), (-2.0, 3.0)),
        (np.array([5.0, 5.0]), (4.5, 5.5)),
        (np.array([], np.int32), (0.0, 1.0)),
    ],
)
def test_default_range(arr, expected):
    assert default_range(arr) == expected


def test_int32_default_range_counts_every_value():
    arr = np.array([[-5, 0, 7]], dtype=np.int32)
    hist = get_hist(arr, bins=13)
    np.testing.assert_array_equal(hist, _oracle(arr, 13, (-5, 8)))
    assert hist.sum() == 3


def test_values_outside_explicit_range_are_ignored():
    arr = np.array([[0, 50, 100, 255]], dtype=np.uint8)
    assert get_hist(arr, bins=10, value_range=(40, 110)).sum() == 2


# ---------------------------------------------------------------- basics carried over from 1.x


def test_hist_matches_bincount(gray_u8):
    hist = get_hist(gray_u8)
    assert hist.dtype == np.int64
    np.testing.assert_array_equal(hist, np.bincount(gray_u8.ravel(), minlength=256))


def test_all_black_and_white():
    assert get_hist(np.zeros((10, 10), np.uint8))[0] == 100
    hist = get_hist(np.full((10, 10), 255, np.uint8))
    assert hist[255] == 100
    assert hist[:255].sum() == 0


def test_empty_array():
    hist = get_hist(np.zeros((0, 0), dtype=np.uint8))
    assert hist.shape == (256,)
    assert hist.sum() == 0


def test_density():
    arr = np.array([[0, 128, 255]] * 3, dtype=np.uint8)
    hist = get_hist(arr, density=True)
    assert hist.dtype == np.float64
    assert hist.sum() == pytest.approx(1.0)
    assert hist[0] == pytest.approx(1 / 3)


def test_density_rgb_rows_each_sum_to_one(rgb_u8):
    hist = get_hist(rgb_u8, channels="rgb", density=True)
    np.testing.assert_allclose(hist.sum(axis=1), 1.0)


def test_density_empty_is_zeros_without_warning():
    hist = get_hist(np.zeros((0, 0), dtype=np.uint8), density=True)
    assert hist.shape == (256,)
    assert not hist.any()


def test_rgba_array():
    rgba = np.zeros((10, 10, 4), dtype=np.uint8)
    rgba[..., 0] = 128
    rgba[..., 3] = 255
    assert get_hist(rgba).sum() == 100


def test_int_and_float_dtypes():
    for dtype in (np.int32, np.float64):
        arr = np.array([[0, 128, 255]] * 3, dtype=dtype)
        hist = get_hist(arr, bins=256, value_range=(0, 256))
        assert (hist[0], hist[128], hist[255]) == (3, 3, 3)


def test_bool_array():
    hist = get_hist(np.array([[True, False], [False, True]]))
    assert (hist[0], hist[255]) == (2, 2)


def test_get_hist_large_image_rgb_to_gray(rng):
    arr = rng.integers(0, 256, size=(300, 400, 3), dtype=np.uint8)
    assert get_hist(arr).sum() == 120_000


# ---------------------------------------------------------------- errors


@pytest.mark.parametrize("bins", [0, -1, 2.5, "10", True, None])
def test_bad_bins(gray_u8, bins):
    with pytest.raises(ValueError, match="bins must be a positive integer"):
        get_hist(gray_u8, bins=bins)


def test_numpy_integer_bins_accepted(gray_u8):
    assert get_hist(gray_u8, bins=np.int64(16)).shape == (16,)


@pytest.mark.parametrize("value_range", [(5, 5), (10, 0), (0, np.inf), (np.nan, 1)])
def test_bad_value_range(gray_u8, value_range):
    with pytest.raises(ValueError, match="value_range must be finite"):
        get_hist(gray_u8, value_range=value_range)


@pytest.mark.parametrize("value_range", [(1,), (1, 2, 3), "ab", 5])
def test_value_range_not_a_pair(gray_u8, value_range):
    with pytest.raises(ValueError, match="value_range must be a"):
        get_hist(gray_u8, value_range=value_range)


def test_rgb_channels_on_gray_raises(gray_u8):
    with pytest.raises(ValueError, match="channels='rgb' requires"):
        get_hist(gray_u8, channels="rgb")


@pytest.mark.parametrize("shape", [(25,), (2, 2, 2, 2)])
def test_bad_shapes(shape):
    with pytest.raises(ValueError, match="must have shape"):
        get_hist(np.zeros(shape, dtype=np.uint8))


def test_nan_raises():
    with pytest.raises(ValueError, match="NaN or infinite"):
        get_hist(np.array([[np.nan]]))
