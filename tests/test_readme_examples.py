"""Run every ```python block in README.md so the documented examples stay correct."""

import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

README = Path(__file__).resolve().parent.parent / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), flags=re.DOTALL)


def test_readme_has_examples():
    assert len(BLOCKS) >= 2


@pytest.mark.parametrize("code", BLOCKS, ids=[f"block{i}" for i in range(len(BLOCKS))])
def test_readme_example_runs(code, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save("photo.jpg")
    exec(compile(code, str(README), "exec"), {"__name__": "__readme__"})
