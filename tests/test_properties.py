"""Property-based tests (hypothesis) for the histogram fast paths and accumulator."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from image_array_and_histogram import HistAccumulator, get_hist

uint_arrays = st.sampled_from([np.uint8, np.uint16]).flatmap(
    lambda dt: arrays(dt, st.tuples(st.integers(0, 12), st.integers(1, 12)))
)


@st.composite
def range_and_bins(draw, dtype):
    top = np.iinfo(dtype).max + 1
    lo = draw(st.integers(0, top - 1))
    hi = draw(st.integers(lo + 1, top))
    divisors = [b for b in range(1, min(hi - lo, 300) + 1) if (hi - lo) % b == 0]
    bins = draw(st.sampled_from(divisors) | st.integers(1, 300))
    return (lo, hi), bins


@settings(max_examples=300, deadline=None)
@given(data=st.data(), arr=uint_arrays)
def test_get_hist_matches_np_histogram(data, arr):
    value_range, bins = data.draw(range_and_bins(arr.dtype))
    np.testing.assert_array_equal(
        get_hist(arr, bins=bins, value_range=value_range),
        np.histogram(arr, bins=bins, range=value_range)[0],
    )


@settings(max_examples=100, deadline=None)
@given(
    imgs=st.lists(
        arrays(np.uint8, st.tuples(st.integers(0, 6), st.integers(1, 6))), min_size=3, max_size=3
    )
)
def test_merge_is_associative(imgs):
    def acc_of(*images):
        acc = HistAccumulator(bins=32)
        for img in images:
            acc.update(img)
        return acc

    a, b, c = (acc_of(i) for i in imgs)
    left = acc_of()
    left.merge(a)
    left.merge(b)
    left.merge(c)
    bc = acc_of()
    bc.merge(b)
    bc.merge(c)
    right = acc_of()
    right.merge(a)
    right.merge(bc)
    np.testing.assert_array_equal(left.counts, right.counts)
    np.testing.assert_array_equal(left.counts, acc_of(*imgs).counts)
