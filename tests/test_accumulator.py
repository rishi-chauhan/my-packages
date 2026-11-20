import numpy as np
import pytest
from PIL import Image

from image_array_and_histogram import HistAccumulator, get_hist


def _images(rng, n=4, dtype=np.uint8, hi=256, shape=(20, 30)):
    return [rng.integers(0, hi, size=shape, dtype=dtype) for _ in range(n)]


def test_update_equals_concatenated_histogram(rng):
    imgs = _images(rng)
    acc = HistAccumulator()
    for img in imgs:
        acc.update(img)
    np.testing.assert_array_equal(acc.counts, get_hist(np.concatenate(imgs)))
    assert acc.n_images == 4
    np.testing.assert_array_equal(acc.edges, np.linspace(0, 256, 257))


def test_merge_equals_single_accumulator(rng):
    imgs = _images(rng, n=6, dtype=np.uint16, hi=65536)
    a, b, total = HistAccumulator(bins=64), HistAccumulator(bins=64), HistAccumulator(bins=64)
    for i, img in enumerate(imgs):
        (a if i % 2 else b).update(img)
        total.update(img)
    a.merge(b)
    np.testing.assert_array_equal(a.counts, total.counts)
    assert a.n_images == 6


def test_merge_is_associative(rng):
    imgs = _images(rng, n=3)
    accs = []
    for img in imgs:
        acc = HistAccumulator()
        acc.update(img)
        accs.append(acc)
    left = HistAccumulator()
    left.merge(accs[0])
    left.merge(accs[1])
    left.merge(accs[2])
    right = HistAccumulator()
    tail = HistAccumulator()
    tail.merge(accs[1])
    tail.merge(accs[2])
    right.merge(accs[0])
    right.merge(tail)
    np.testing.assert_array_equal(left.counts, right.counts)


def test_merge_empty_is_noop(rng):
    acc = HistAccumulator()
    acc.update(_images(rng, n=1)[0])
    before = acc.counts.copy()
    acc.merge(HistAccumulator())
    np.testing.assert_array_equal(acc.counts, before)
    assert acc.n_images == 1


def test_rgb_accumulator(rgb_u8):
    acc = HistAccumulator(channels="rgb")
    acc.update(rgb_u8)
    acc.update(Image.fromarray(rgb_u8))
    np.testing.assert_array_equal(acc.counts, 2 * get_hist(rgb_u8, channels="rgb"))
    stats = acc.stats()
    assert set(stats) == {"r", "g", "b"}
    assert acc.channels == "rgb"


def test_float_images_with_explicit_range(gray_f32):
    acc = HistAccumulator(bins=10, value_range=(0, 1))
    acc.update(gray_f32)
    acc.update(gray_f32)
    np.testing.assert_array_equal(acc.counts, 2 * get_hist(gray_f32, bins=10, value_range=(0, 1)))
    assert acc.value_range == (0.0, 1.0)
    assert acc.bins == 10


def test_counts_is_read_only(gray_u8):
    acc = HistAccumulator()
    acc.update(gray_u8)
    with pytest.raises(ValueError, match="read-only"):
        acc.counts[0] = 5


def test_stats(gray_u8):
    acc = HistAccumulator()
    acc.update(gray_u8)
    assert acc.stats().mean == pytest.approx(gray_u8.mean())


def test_repr():
    assert "HistAccumulator(channels='gray', bins=256" in repr(HistAccumulator())


# ---------------------------------------------------------------- errors


def test_float_without_range_raises(gray_f32):
    with pytest.raises(ValueError, match="need an explicit value_range"):
        HistAccumulator().update(gray_f32)


def test_int32_without_range_raises():
    with pytest.raises(ValueError, match="need an explicit value_range"):
        HistAccumulator().update(np.zeros((2, 2), dtype=np.int32))


def test_mixed_bit_depths_raise(gray_u8, gray_u16):
    acc = HistAccumulator()
    acc.update(gray_u8)
    with pytest.raises(ValueError, match="pass value_range explicitly"):
        acc.update(gray_u16)
    assert acc.n_images == 1


def test_mixed_bit_depths_with_explicit_range(gray_u8, gray_u16):
    acc = HistAccumulator(bins=16, value_range=(0, 65536))
    acc.update(gray_u8)
    acc.update(gray_u16)
    assert acc.counts.sum() == gray_u8.size + gray_u16.size


def test_merge_mismatches(gray_u8, gray_u16):
    with pytest.raises(TypeError, match="can only merge"):
        HistAccumulator().merge(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="different channels or bins"):
        HistAccumulator().merge(HistAccumulator(bins=8))
    a, b = HistAccumulator(), HistAccumulator()
    a.update(gray_u8)
    b.update(gray_u16)
    with pytest.raises(ValueError, match="different ranges"):
        a.merge(b)


def test_empty_accumulator():
    acc = HistAccumulator()
    assert acc.counts.sum() == 0
    assert acc.value_range is None
    with pytest.raises(ValueError, match="edges are unknown"):
        _ = acc.edges
    with pytest.raises(ValueError, match="empty histogram"):
        acc.stats()


def test_bad_config():
    with pytest.raises(ValueError, match="bins must be"):
        HistAccumulator(bins=0)
    with pytest.raises(ValueError, match="channels must be"):
        HistAccumulator(channels="hsv")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="value_range must be finite"):
        HistAccumulator(value_range=(1, 1))
