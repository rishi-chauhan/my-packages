import numpy as np
import pytest
from PIL import Image

from image_array_and_histogram import array_to_image, get_image_array
from image_array_and_histogram._convert import _to_array

# ---------------------------------------------------------------- get_image_array


def test_round_trip_array_to_image():
    arr = np.arange(100, dtype=np.uint8).reshape(10, 10)
    back = get_image_array(array_to_image(arr))
    assert back.shape == arr.shape
    np.testing.assert_array_equal(back, arr)


def test_large_image_round_trip(rng):
    arr = rng.integers(0, 256, size=(1000, 1000), dtype=np.uint8)
    img = array_to_image(arr)
    assert img.size == (1000, 1000)
    np.testing.assert_array_equal(get_image_array(img), arr)


def test_single_pixel_image():
    arr = get_image_array(Image.new("L", (1, 1), color=128))
    assert arr.shape == (1, 1)
    assert arr[0, 0] == 128


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "P", "1", "LA", "CMYK", "YCbCr"])
def test_get_image_array_gray_converts_colour_modes(mode):
    img = Image.new("RGB", (4, 3), (200, 10, 30)).convert(mode)
    arr = get_image_array(img)
    assert arr.shape == (3, 4)
    assert arr.dtype == np.uint8
    np.testing.assert_array_equal(arr, np.asarray(img.convert("L")))


def test_get_image_array_rgb_mode():
    img = Image.new("RGBA", (4, 3), (200, 10, 30, 7))
    arr = get_image_array(img, mode="rgb")
    assert arr.shape == (3, 4, 3)
    assert tuple(arr[0, 0]) == (200, 10, 30)


def test_get_image_array_keep_mode():
    img = Image.new("RGBA", (4, 3), (1, 2, 3, 4))
    arr = get_image_array(img, mode="keep")
    assert arr.shape == (3, 4, 4)


def test_get_image_array_keeps_16_bit():
    data = np.array([[1, 1000], [40000, 65535]], dtype=np.uint16)
    arr = get_image_array(Image.fromarray(data))
    assert arr.dtype == np.uint16
    np.testing.assert_array_equal(arr, data)


def test_get_image_array_big_endian_16_bit_is_native():
    data = np.array([[1, 1000], [40000, 65535]], dtype=np.uint16)
    img = Image.frombytes("I;16B", (2, 2), data.astype(">u2").tobytes())
    arr = get_image_array(img)
    assert arr.dtype == np.dtype("=u2")
    np.testing.assert_array_equal(arr, data)


@pytest.mark.parametrize(("mode", "dtype"), [("I", np.int32), ("F", np.float32)])
def test_get_image_array_keeps_int32_and_float(mode, dtype):
    img = Image.fromarray(np.array([[1, 300]], dtype=np.uint16)).convert(mode)
    arr = get_image_array(img)
    assert arr.dtype == dtype
    np.testing.assert_array_equal(arr, [[1, 300]])


@pytest.mark.parametrize("mode", ["I;16", "I", "F"])
def test_get_image_array_rgb_on_single_band_raises(mode):
    img = Image.fromarray(np.zeros((2, 2), dtype=np.uint16)).convert(mode)
    with pytest.raises(ValueError, match="requires a colour image"):
        get_image_array(img, mode="rgb")


def test_get_image_array_rejects_arrays_and_bad_mode():
    with pytest.raises(TypeError, match=r"PIL\.Image\.Image"):
        get_image_array(np.zeros((2, 2)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mode must be"):
        get_image_array(Image.new("L", (1, 1)), mode="grey")  # type: ignore[arg-type]


# ---------------------------------------------------------------- _to_array


def test_rgb_uint8_array_gray_matches_pil_image(rgb_u8):
    from_array = _to_array(rgb_u8, "gray")
    from_pil = _to_array(Image.fromarray(rgb_u8), "gray")
    np.testing.assert_array_equal(from_array, from_pil)


def test_rgba_array_drops_alpha(rgb_u8):
    rgba = np.dstack([rgb_u8, np.full(rgb_u8.shape[:2], 7, np.uint8)])
    np.testing.assert_array_equal(_to_array(rgba, "gray"), _to_array(rgb_u8, "gray"))
    np.testing.assert_array_equal(_to_array(rgba, "rgb"), rgb_u8)


def test_rgb_uint16_array_to_gray_keeps_dtype():
    rgb = np.zeros((1, 2, 3), dtype=np.uint16)
    rgb[0, 0] = (65535, 65535, 65535)
    rgb[0, 1] = (1000, 0, 0)
    gray = _to_array(rgb, "gray")
    assert gray.dtype == np.uint16
    np.testing.assert_array_equal(gray, [[65535, 299]])


def test_rgb_float_and_wide_int_arrays_to_gray():
    f = np.array([[[1.0, 0.5, 0.0]]], dtype=np.float64)
    out = _to_array(f, "gray")
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [[0.299 + 0.5 * 0.587]])
    i = np.array([[[100_000_000, 0, 0]]], dtype=np.int64)
    assert _to_array(i, "gray")[0, 0] == 29_900_000


def test_empty_rgb_array_to_gray():
    out = _to_array(np.zeros((0, 3, 3), dtype=np.uint8), "gray")
    assert out.shape == (0, 3)


def test_to_array_bool_maps_to_255():
    out = _to_array(np.array([[True, False]]), "gray")
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, [[255, 0]])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_to_array_rejects_nan_inf(bad):
    with pytest.raises(ValueError, match="NaN or infinite"):
        _to_array(np.array([[1.0, bad]]), "gray")


@pytest.mark.parametrize("arr", [np.array([["a"]]), np.array([[1 + 2j]]), np.array([[None]])])
def test_to_array_rejects_non_numeric(arr):
    with pytest.raises(TypeError, match="numeric"):
        _to_array(arr, "gray")


@pytest.mark.parametrize("shape", [(5,), (2, 2, 2), (2, 2, 5), (1, 1, 1, 1)])
def test_to_array_rejects_bad_shapes(shape):
    with pytest.raises(ValueError, match="must have shape"):
        _to_array(np.zeros(shape), "gray")


def test_to_array_rgb_on_2d_array_raises():
    with pytest.raises(ValueError, match="channels='rgb' requires"):
        _to_array(np.zeros((2, 2)), "rgb")


def test_to_array_bad_channels():
    with pytest.raises(ValueError, match="channels must be"):
        _to_array(np.zeros((2, 2)), "hsv")  # type: ignore[arg-type]


# ---------------------------------------------------------------- array_to_image


def test_array_to_image_flat_input():
    img = array_to_image(list(range(25)), width=5, height=5)
    assert img.size == (5, 5)
    assert get_image_array(img).shape == (5, 5)


@pytest.mark.parametrize(("channels", "mode"), [(3, "RGB"), (4, "RGBA")])
def test_array_to_image_colour(channels, mode):
    arr = np.arange(2 * 3 * channels, dtype=np.uint8).reshape(2, 3, channels)
    img = array_to_image(arr)
    assert img.mode == mode
    assert img.size == (3, 2)
    np.testing.assert_array_equal(np.asarray(img), arr)


def test_array_to_image_uint16_is_lossless():
    data = np.array([[1, 1000], [40000, 65535]], dtype=np.uint16)
    img = array_to_image(data)
    assert img.mode == "I;16"
    np.testing.assert_array_equal(np.asarray(img), data)


def test_array_to_image_mode_l_for_other_dtypes():
    assert array_to_image(np.array([[1.0, 2.0]])).mode == "L"
    assert array_to_image(np.array([[1, 2]], dtype=np.int64)).mode == "L"


def test_float_values_are_rounded_not_truncated():
    img = array_to_image(np.array([[0.0, 127.5], [200.7, 255.0], [127.9, 127.4]]))
    np.testing.assert_array_equal(get_image_array(img), [[0, 128], [201, 255], [128, 127]])


def test_bool_array_maps_to_0_and_255():
    img = array_to_image(np.array([[True, False], [False, True]]))
    np.testing.assert_array_equal(get_image_array(img), [[255, 0], [0, 255]])


def test_clip_true_warns_and_points_to_caller():
    with pytest.warns(RuntimeWarning, match="clipped") as record:
        img = array_to_image(np.array([[300, -20]]))
    assert record[0].filename == __file__
    np.testing.assert_array_equal(get_image_array(img), [[255, 0]])


def test_clip_false_raises_on_out_of_range():
    with pytest.raises(ValueError, match=r"arr values must be in \[0, 255\]"):
        array_to_image(np.array([[300]]), clip=False)
    img = array_to_image(np.array([[0, 255]]), clip=False)
    np.testing.assert_array_equal(get_image_array(img), [[0, 255]])


def test_array_to_image_uint16_rgb_is_clipped():
    with pytest.warns(RuntimeWarning, match="clipped"):
        img = array_to_image(np.full((1, 1, 3), 1000, dtype=np.uint16))
    assert img.mode == "RGB"


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_array_to_image_nan_inf_raise(bad):
    with pytest.raises(ValueError, match="NaN or infinite"):
        array_to_image(np.array([[1.0, bad]]))


def test_array_to_image_non_numeric_raises():
    with pytest.raises(TypeError, match="numeric"):
        array_to_image(np.array([["a", "b"]]))


def test_array_to_image_missing_dimensions():
    for kwargs in ({}, {"width": 5}, {"height": 5}):
        with pytest.raises(ValueError, match="width and height are required"):
            array_to_image(list(range(25)), **kwargs)


def test_array_to_image_dimension_mismatch():
    with pytest.raises(ValueError, match=r"does not equal width\*height"):
        array_to_image(list(range(25)), width=4, height=5)


def test_array_to_image_2d_dimension_validation():
    arr = np.arange(20, dtype=np.uint8).reshape(4, 5)
    with pytest.raises(ValueError, match=r"Provided width .* does not match"):
        array_to_image(arr, width=10)
    with pytest.raises(ValueError, match=r"Provided height .* does not match"):
        array_to_image(arr, height=10)
    assert array_to_image(arr, width=5, height=4).size == (5, 4)


@pytest.mark.parametrize("shape", [(2, 2, 2), (2, 2, 5), (1, 1, 1, 1)])
def test_array_to_image_bad_shapes(shape):
    with pytest.raises(ValueError, match="arr must have shape"):
        array_to_image(np.zeros(shape, dtype=np.uint8))


def test_array_to_image_empty_raises_clear_error():
    for arr in (np.zeros((0, 0)), np.zeros((0, 3)), np.zeros((0, 2, 3), dtype=np.uint8)):
        with pytest.raises(ValueError, match="at least one pixel"):
            array_to_image(arr)


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [({"width": 0, "height": 5}, "width"), ({"width": 5, "height": -1}, "height")],
)
def test_array_to_image_non_positive_size(kwargs, name):
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        array_to_image([], **kwargs)
