import warnings

import numpy as np
import pytest
from PIL import Image

from image_array_and_histogram import (
    array_to_image,
    get_hist,
    get_image_array,
    getHist,
    getImageArray,
    getImageFromArray,
)


def test_round_trip_array_to_image():
    arr = np.arange(100, dtype=np.uint8).reshape(10, 10)
    img = array_to_image(arr)
    back = get_image_array(img)
    assert back.shape == arr.shape
    assert np.array_equal(back, arr)


def test_hist_matches_numpy():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(32, 48), dtype=np.uint8)
    expected = np.bincount(arr.ravel(), minlength=256)[:256]
    hist = get_hist(arr)
    assert hist == expected.tolist()


def test_accept_rgb_image():
    # Create an RGB gradient image
    data = np.zeros((8, 8, 3), dtype=np.uint8)
    data[..., 0] = np.arange(64, dtype=np.uint8).reshape(8, 8)
    img = Image.fromarray(data, mode="RGB")
    arr = get_image_array(img)  # implicit grayscale conversion
    assert arr.shape == (8, 8)
    hist = get_hist(img)
    assert sum(hist) == 64


def test_array_to_image_flat_input():
    arr = list(range(25))
    img = array_to_image(arr, width=5, height=5)
    assert img.size == (5, 5)
    back = get_image_array(img)
    assert back.shape == (5, 5)


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_single_pixel_image():
    """Test handling of single-pixel images."""
    img = Image.new("L", (1, 1), color=128)
    arr = get_image_array(img)
    assert arr.shape == (1, 1)
    assert arr[0, 0] == 128

    hist = get_hist(arr)
    assert hist[128] == 1
    assert sum(hist) == 1


def test_empty_array_histogram():
    """Test histogram with empty array."""
    # Empty 0x0 array
    arr = np.array([], dtype=np.uint8).reshape(0, 0)
    hist = get_hist(arr)
    assert len(hist) == 256
    assert sum(hist) == 0


def test_large_image():
    """Test handling of large images."""
    # Create a large 1000x1000 image
    large_arr = np.random.default_rng(0).integers(0, 256, size=(1000, 1000), dtype=np.uint8)
    img = array_to_image(large_arr)
    assert img.size == (1000, 1000)

    back = get_image_array(img)
    assert np.array_equal(back, large_arr)


def test_all_black_image():
    """Test image with all black pixels."""
    arr = np.zeros((10, 10), dtype=np.uint8)
    hist = get_hist(arr)
    assert hist[0] == 100
    assert sum(hist[1:]) == 0


def test_all_white_image():
    """Test image with all white pixels."""
    arr = np.full((10, 10), 255, dtype=np.uint8)
    hist = get_hist(arr)
    assert hist[255] == 100
    assert sum(hist[:255]) == 0


def test_histogram_as_density():
    """Test histogram with density normalization."""
    arr = np.array([[0, 128, 255]] * 3, dtype=np.uint8)  # 3x3 with 3 values each
    hist = get_hist(arr, as_density=True)

    assert len(hist) == 256
    assert abs(sum(hist) - 1.0) < 1e-10  # Should sum to 1.0
    assert abs(hist[0] - 1 / 3) < 1e-10
    assert abs(hist[128] - 1 / 3) < 1e-10
    assert abs(hist[255] - 1 / 3) < 1e-10


def test_histogram_density_empty():
    """Test density histogram with empty array."""
    arr = np.array([], dtype=np.uint8).reshape(0, 0)
    hist = get_hist(arr, as_density=True)
    assert len(hist) == 256
    assert all(h == 0.0 for h in hist)


# ============================================================================
# Error Condition Tests
# ============================================================================


def test_array_to_image_missing_dimensions():
    """Test that flat array without dimensions raises error."""
    arr = list(range(25))

    with pytest.raises(ValueError, match="width and height are required"):
        array_to_image(arr)

    with pytest.raises(ValueError, match="width and height are required"):
        array_to_image(arr, width=5)

    with pytest.raises(ValueError, match="width and height are required"):
        array_to_image(arr, height=5)


def test_array_to_image_dimension_mismatch():
    """Test that mismatched dimensions raise error."""
    arr = list(range(25))

    with pytest.raises(ValueError, match="does not equal width\\*height"):
        array_to_image(arr, width=4, height=5)

    with pytest.raises(ValueError, match="does not equal width\\*height"):
        array_to_image(arr, width=5, height=6)


def test_array_to_image_2d_dimension_validation():
    """Test dimension validation for 2D arrays."""
    arr = np.arange(20, dtype=np.uint8).reshape(4, 5)

    # Mismatched width
    with pytest.raises(ValueError, match=r"Provided width .* does not match"):
        array_to_image(arr, width=10)

    # Mismatched height
    with pytest.raises(ValueError, match=r"Provided height .* does not match"):
        array_to_image(arr, height=10)


def test_array_to_image_3d_array():
    """Test that 3D arrays raise error."""
    arr = np.zeros((5, 5, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Only 1D or 2D arrays supported"):
        array_to_image(arr)


def test_get_image_array_converts_rgb_to_2d():
    """RGB images are converted to a 2D grayscale array."""
    # Create a 3D array and try to pass it as grayscale
    arr_3d = np.zeros((5, 5, 3), dtype=np.uint8)
    img = Image.fromarray(arr_3d, mode="RGB")

    # This should work due to automatic conversion
    result = get_image_array(img)
    assert result.ndim == 2


def test_get_hist_non_2d_array():
    """Test that non-2D non-RGB arrays raise error."""
    arr_1d = np.zeros(25, dtype=np.uint8)

    with pytest.raises(ValueError, match="Expected a 2D array"):
        get_hist(arr_1d)

    arr_4d = np.zeros((2, 2, 2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="Expected a 2D array"):
        get_hist(arr_4d)


def test_array_to_image_out_of_range_values():
    """Test handling of out-of-range pixel values."""
    # Values above 255
    arr = np.array([[300, 400], [500, 600]])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        img = array_to_image(arr)
        # Check that we got at least one warning about clipping
        clip_warnings = [warning for warning in w if "clipped" in str(warning.message).lower()]
        assert len(clip_warnings) >= 1

    back = get_image_array(img)
    assert back.max() == 255

    # Negative values
    arr = np.array([[-10, -20], [-30, 100]])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        img = array_to_image(arr)
        # Check that we got at least one warning about clipping
        clip_warnings = [warning for warning in w if "clipped" in str(warning.message).lower()]
        assert len(clip_warnings) >= 1

    back = get_image_array(img)
    assert back.min() == 0


def test_rgb_to_grayscale_conversion():
    """Test RGB image conversion to grayscale."""
    # Create RGB image
    rgb_arr = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb_arr[:, :, 0] = 255  # Red channel
    img = Image.fromarray(rgb_arr, mode="RGB")

    # Should convert to grayscale automatically
    gray_arr = get_image_array(img, ensure_grayscale=True)
    assert gray_arr.ndim == 2
    assert gray_arr.shape == (10, 10)


def test_rgba_to_grayscale_conversion():
    """Test RGBA image conversion to grayscale."""
    # Create RGBA image
    rgba_arr = np.zeros((10, 10, 4), dtype=np.uint8)
    rgba_arr[:, :, 0] = 255  # Red channel
    rgba_arr[:, :, 3] = 255  # Alpha channel
    img = Image.fromarray(rgba_arr, mode="RGBA")

    # Should convert to grayscale automatically
    gray_arr = get_image_array(img, ensure_grayscale=True)
    assert gray_arr.ndim == 2
    assert gray_arr.shape == (10, 10)


def test_get_hist_with_rgb_array():
    """Test histogram computation with RGB array directly."""
    rgb_arr = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb_arr[:, :, 0] = 128  # Red channel

    hist = get_hist(rgb_arr)
    assert len(hist) == 256
    assert sum(hist) == 100  # 10x10 pixels


def test_get_hist_with_rgba_array():
    """Test histogram computation with RGBA array directly."""
    rgba_arr = np.zeros((10, 10, 4), dtype=np.uint8)
    rgba_arr[:, :, 0] = 128  # Red channel
    rgba_arr[:, :, 3] = 255  # Alpha channel

    hist = get_hist(rgba_arr)
    assert len(hist) == 256
    assert sum(hist) == 100  # 10x10 pixels


# ============================================================================
# Deprecated Function Tests
# ============================================================================


def test_deprecated_getImageArray():
    """Test deprecated getImageArray function."""
    img = Image.new("L", (10, 10), color=128)

    with pytest.warns(DeprecationWarning, match="getImageArray is deprecated"):
        arr = getImageArray(img)

    assert arr.shape == (10, 10)
    assert np.all(arr == 128)


def test_deprecated_getHist():
    """Test deprecated getHist function."""
    arr = np.full((10, 10), 128, dtype=np.uint8)

    with pytest.warns(DeprecationWarning, match="getHist is deprecated"):
        hist = getHist(arr)

    assert len(hist) == 256
    assert hist[128] == 100


def test_deprecated_getImageFromArray():
    """Test deprecated getImageFromArray function."""
    arr = list(range(25))

    with pytest.warns(DeprecationWarning, match="getImageFromArray is deprecated"):
        img = getImageFromArray(arr, 5, 5)

    assert img.size == (5, 5)


# ============================================================================
# Type Coercion Tests
# ============================================================================


def test_array_to_image_float_coercion():
    """Test that float arrays are properly converted to uint8."""
    arr = np.array([[0.0, 127.5], [200.7, 255.0]])
    img = array_to_image(arr)
    back = get_image_array(img)

    # Should be rounded (not truncated) and converted to uint8
    assert back.dtype == np.uint8
    np.testing.assert_array_equal(back, [[0, 128], [201, 255]])


def test_get_hist_with_different_dtypes():
    """Test histogram with different array dtypes."""
    # int32 array
    arr_int32 = np.array([[0, 128, 255]] * 3, dtype=np.int32)
    hist = get_hist(arr_int32)
    assert hist[0] == 3
    assert hist[128] == 3
    assert hist[255] == 3

    # float64 array (will be clipped and converted)
    arr_float = np.array([[0.0, 128.0, 255.0]] * 3, dtype=np.float64)
    hist = get_hist(arr_float)
    assert hist[0] == 3
    assert hist[128] == 3
    assert hist[255] == 3


# ============================================================================
# 1.1.9 bug-fix regression tests
# ============================================================================


def test_validate_range_false_raises_on_out_of_range():
    with pytest.raises(ValueError, match=r"arr values must be in \[0, 255\]"):
        array_to_image(np.array([[300]]), validate_range=False)
    with pytest.raises(ValueError, match=r"arr values must be in \[0, 255\]"):
        array_to_image(np.array([[-1, 5]]), validate_range=False)
    # In-range values are fine
    img = array_to_image(np.array([[0, 255]]), validate_range=False)
    np.testing.assert_array_equal(get_image_array(img), [[0, 255]])


def test_get_hist_rgb_array_out_of_range_does_not_wrap():
    with pytest.warns(RuntimeWarning, match="clipped"):
        hist = get_hist(np.array([[[300, 300, 300]]]))
    assert hist[255] == 1
    assert hist[44] == 0


def test_get_hist_gray_out_of_range_warns():
    with pytest.warns(RuntimeWarning, match="clipped"):
        hist = get_hist(np.array([[300, -5]]))
    assert hist[255] == 1
    assert hist[0] == 1


def test_get_hist_in_range_int_array_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hist = get_hist(np.array([[0, 255]], dtype=np.int64))
    assert hist[0] == 1
    assert hist[255] == 1


def test_array_to_image_empty_raises_clear_error():
    with pytest.raises(ValueError, match="at least one pixel"):
        array_to_image(np.zeros((0, 0)))
    with pytest.raises(ValueError, match="at least one pixel"):
        array_to_image(np.zeros((0, 3)))
    for kwargs in ({"width": 0, "height": 5}, {"width": 5, "height": -1}):
        name = "width" if kwargs["width"] <= 0 else "height"
        with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
            array_to_image([], **kwargs)


def test_array_to_image_empty_flat_without_dims_raises():
    with pytest.raises(ValueError, match="width and height are required"):
        array_to_image([])


def test_float_values_are_rounded_not_truncated():
    img = array_to_image(np.array([[127.9, 127.4, 0.5, 1.5]]))
    np.testing.assert_array_equal(get_image_array(img), [[128, 127, 0, 2]])
    hist = get_hist(np.array([[127.9]]))
    assert hist[128] == 1


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nan_and_inf_raise(bad):
    arr = np.array([[1.0, bad]])
    with pytest.raises(ValueError, match="NaN or infinite"):
        get_hist(arr)
    with pytest.raises(ValueError, match="NaN or infinite"):
        array_to_image(arr)


def test_non_numeric_dtype_raises_type_error():
    with pytest.raises(TypeError, match="must be numeric"):
        array_to_image(np.array([["a", "b"]]))
    with pytest.raises(TypeError, match="must be numeric"):
        get_hist(np.array([[1 + 2j]]))


def test_bool_array_maps_to_0_and_255():
    mask = np.array([[True, False], [False, True]])
    img = array_to_image(mask)
    np.testing.assert_array_equal(get_image_array(img), [[255, 0], [0, 255]])
    hist = get_hist(mask)
    assert hist[0] == 2
    assert hist[255] == 2


def test_warning_points_to_caller():
    with pytest.warns(RuntimeWarning) as record:
        array_to_image(np.array([[300]]))
    assert record[0].filename == __file__
    with pytest.warns(RuntimeWarning) as record:
        get_hist(np.array([[300]]))
    assert record[0].filename == __file__


def test_get_image_array_16bit_without_grayscale_clips_with_warning():
    img = Image.fromarray(np.array([[10, 1000]], dtype=np.uint16))
    with pytest.warns(RuntimeWarning, match="clipped"):
        arr = get_image_array(img, ensure_grayscale=False)
    assert arr.dtype == np.uint8
    np.testing.assert_array_equal(arr, [[10, 255]])


def test_array_to_image_returns_mode_l():
    img = array_to_image(np.array([[1.0, 2.0]]))
    assert img.mode == "L"


def test_density_returns_python_floats():
    hist = get_hist(np.array([[0, 1]], dtype=np.uint8), as_density=True)
    assert isinstance(hist, list)
    assert isinstance(hist[0], float)


def test_version_matches_installed_metadata():
    import importlib.metadata

    import image_array_and_histogram

    assert image_array_and_histogram.__version__ == importlib.metadata.version(
        "image_array_and_histogram"
    )


def test_get_hist_empty_float_array():
    hist = get_hist(np.zeros((0, 0), dtype=np.float64))
    assert sum(hist) == 0


def test_get_image_array_colour_without_grayscale_raises():
    img = Image.new("RGB", (2, 2))
    with pytest.raises(ValueError, match="Expected a 2D grayscale image"):
        get_image_array(img, ensure_grayscale=False)
