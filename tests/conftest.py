import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def gray_u8(rng):
    return rng.integers(0, 256, size=(32, 48), dtype=np.uint8)


@pytest.fixture
def rgb_u8(rng):
    return rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)


@pytest.fixture
def gray_u16(rng):
    return rng.integers(0, 65536, size=(32, 48), dtype=np.uint16)


@pytest.fixture
def gray_f32(rng):
    return rng.random(size=(32, 48), dtype=np.float32)
