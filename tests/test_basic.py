import numpy as np
from PIL import Image

from image_array_and_histogram import (
    get_image_array,
    get_hist,
    array_to_image,
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
