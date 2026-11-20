import importlib.metadata

import image_array_and_histogram as pkg


def test_version_matches_installed_metadata():
    assert pkg.__version__ == importlib.metadata.version("image_array_and_histogram")


def test_public_api():
    assert set(pkg.__all__) == {
        "get_image_array", "array_to_image", "get_hist", "hist_stats", "stats_from_counts",
        "compare_hist", "HistAccumulator", "HistStats", "__version__",
    }  # fmt: skip
    for name in pkg.__all__:
        assert hasattr(pkg, name)


def test_removed_1x_api():
    for name in ("getImageArray", "getHist", "getImageFromArray"):
        assert not hasattr(pkg, name)


def test_import_does_not_load_matplotlib():
    import subprocess
    import sys

    code = "import sys, image_array_and_histogram; print('matplotlib' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
