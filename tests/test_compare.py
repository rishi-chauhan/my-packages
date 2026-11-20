import numpy as np
import pytest

from image_array_and_histogram import compare_hist, get_hist
from image_array_and_histogram._compare import METHOD_INFO

METHODS = list(METHOD_INFO)


@pytest.mark.parametrize("method", METHODS)
def test_identical_inputs_give_identical_value(gray_u8, method):
    h = get_hist(gray_u8)
    assert compare_hist(h, h, method=method) == pytest.approx(METHOD_INFO[method][1])


@pytest.mark.parametrize("method", METHODS)
def test_scale_invariant_except_correlation(gray_u8, method):
    h = get_hist(gray_u8)
    expected = METHOD_INFO[method][1]
    assert compare_hist(h, h * 3, method=method) == pytest.approx(expected)


@pytest.mark.parametrize("method", METHODS)
def test_symmetry(rng, method):
    a = rng.integers(0, 100, 64)
    b = rng.integers(0, 100, 64)
    assert compare_hist(a, b, method=method) == pytest.approx(compare_hist(b, a, method=method))


def test_disjoint_histograms():
    a = np.array([5, 0, 0, 0])
    b = np.array([0, 0, 0, 7])
    assert compare_hist(a, b, method="intersection") == 0.0
    assert compare_hist(a, b, method="chi2") == pytest.approx(1.0)
    assert compare_hist(a, b, method="bhattacharyya") == pytest.approx(1.0)
    assert compare_hist(a, b, method="emd") == pytest.approx(3.0)


def test_known_values():
    a = np.array([1, 1, 0, 0])
    b = np.array([0, 1, 1, 0])
    assert compare_hist(a, b) == pytest.approx(0.5)
    assert compare_hist(a, b, method="emd") == pytest.approx(1.0)
    assert compare_hist(a, b, method="chi2") == pytest.approx(0.5)
    assert compare_hist(a, b, method="bhattacharyya") == pytest.approx(np.sqrt(0.5))
    assert compare_hist([1, 2, 3], [3, 2, 1], method="correlation") == pytest.approx(-1.0)


def test_correlation_constant_histograms():
    assert compare_hist([2, 2, 2], [2, 2, 2], method="correlation") == 1.0
    assert compare_hist([2, 2, 2], [1, 2, 3], method="correlation") == 0.0
    assert compare_hist([0, 0, 0], [0, 0, 0], method="correlation") == 1.0


def test_2d_input_is_mean_of_rows(rgb_u8, rng):
    h1 = get_hist(rgb_u8, channels="rgb")
    h2 = get_hist(rng.integers(0, 256, rgb_u8.shape, dtype=np.uint8), channels="rgb")
    rows = [compare_hist(h1[i], h2[i], method="chi2") for i in range(3)]
    assert compare_hist(h1, h2, method="chi2") == pytest.approx(np.mean(rows))


def test_returns_python_float(gray_u8):
    h = get_hist(gray_u8)
    assert type(compare_hist(h, h)) is float


# ---------------------------------------------------------------- errors


def test_shape_mismatch():
    with pytest.raises(ValueError, match="shapes differ"):
        compare_hist(np.ones(4), np.ones(5))


@pytest.mark.parametrize("bad", [np.ones((2, 2, 2)), np.ones(0), np.ones((3, 0))])
def test_bad_shape(bad):
    with pytest.raises(ValueError, match="non-empty 1-D or 2-D"):
        compare_hist(bad, bad)


@pytest.mark.parametrize("bad", [[1, -1, 2], [1, np.nan, 2], [1, np.inf, 2]])
def test_negative_or_non_finite(bad):
    with pytest.raises(ValueError, match="finite, non-negative"):
        compare_hist(bad, [1, 1, 1])


def test_non_numeric():
    with pytest.raises(TypeError, match="h2 must be numeric"):
        compare_hist([1, 2], ["a", "b"])


def test_all_zero_histogram():
    with pytest.raises(ValueError, match="all-zero"):
        compare_hist([0, 0], [1, 1])


def test_unknown_method():
    with pytest.raises(ValueError, match="method must be one of"):
        compare_hist([1], [1], method="kl")  # type: ignore[arg-type]
