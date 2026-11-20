import csv
import io
import json
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from image_array_and_histogram import HistAccumulator
from image_array_and_histogram._report import BLOCKS, sparkline
from image_array_and_histogram.cli import _discover, _exts, main


@pytest.fixture
def images(tmp_path, rng):
    """A small folder: gray, RGB, RGBA PNGs, a nested dir, a hidden file and a corrupt file."""
    d = tmp_path / "imgs"
    (d / "sub").mkdir(parents=True)
    (d / ".hidden").mkdir()
    Image.fromarray(rng.integers(0, 256, (8, 10), np.uint8)).save(d / "a_gray.png")
    Image.fromarray(rng.integers(0, 256, (6, 7, 3), np.uint8)).save(d / "b_rgb.PNG")
    Image.fromarray(rng.integers(0, 256, (5, 5, 4), np.uint8)).save(d / "c_rgba.png")
    Image.fromarray(rng.integers(0, 256, (4, 4), np.uint8)).save(d / "sub" / "d_nested.jpg")
    Image.fromarray(rng.integers(0, 256, (3, 3), np.uint8)).save(d / ".e_hidden.png")
    Image.fromarray(rng.integers(0, 256, (3, 3), np.uint8)).save(d / ".hidden" / "f.png")
    (d / "notes.txt").write_text("not an image")
    return d


@pytest.fixture
def corrupt(tmp_path, rng):
    p = tmp_path / "corrupt.png"
    p.write_bytes(rng.bytes(64))
    return p


def _pixels(paths):
    total = 0
    for p in paths:
        with Image.open(p) as img:
            total += img.width * img.height
    return total


def run(capsys, *argv):
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


# ---------------------------------------------------------------- discovery


def test_discover_non_recursive_skips_hidden_and_other_extensions(images):
    found = _discover([str(images)], recursive=False, exts=_exts(".png,.jpg"))
    assert [p.name for p in found] == ["a_gray.png", "b_rgb.PNG", "c_rgba.png"]


def test_discover_recursive(images):
    found = _discover([str(images)], recursive=True, exts=_exts("png,jpg"))
    assert [p.name for p in found] == ["a_gray.png", "b_rgb.PNG", "c_rgba.png", "d_nested.jpg"]


def test_discover_ext_filter_and_dedupe(images):
    found = _discover(
        [str(images), str(images / "sub" / "d_nested.jpg")], recursive=True, exts=_exts("jpg")
    )
    assert [p.name for p in found] == ["d_nested.jpg"]


def test_discover_explicit_file_always_included(images):
    found = _discover([str(images / "notes.txt")], recursive=False, exts=_exts("png"))
    assert [p.name for p in found] == ["notes.txt"]


# ---------------------------------------------------------------- stats


def test_stats_single_file_text(capsys, images):
    code, out, _ = run(capsys, "stats", str(images / "a_gray.png"))
    assert code == 0
    assert out.startswith("imghist: 1 image (0 skipped) · channel: gray · 256 bins")
    assert "count     80" in out
    assert any(ch in out for ch in BLOCKS)
    assert "\x1b" not in out  # never any ANSI codes


def test_stats_directory_text(capsys, images):
    code, out, _ = run(capsys, "stats", str(images))
    assert code == 0
    assert "3 images (0 skipped)" in out


def test_stats_recursive(capsys, images):
    code, out, _ = run(capsys, "stats", str(images), "-r")
    assert code == 0
    assert "4 images" in out


def test_stats_rgb_text(capsys, images):
    code, out, _ = run(capsys, "stats", str(images), "--channels", "rgb", "--width", "16")
    assert code == 0
    for header in ("\nR\n", "\nG\n", "\nB\n"):
        assert header in out


def test_stats_json_counts_sum_to_pixels(capsys, images):
    code, out, _ = run(capsys, "stats", str(images), "--format", "json", "--per-file")
    assert code == 0
    data = json.loads(out)
    files = sorted(images.glob("[!.]*.[pP][nN][gG]"))
    assert sum(data["counts"]) == _pixels(files)
    assert data["images"] == 3
    assert data["range"] == [0.0, 256.0]
    assert [f["path"] for f in data["per_file"]] == [str(f) for f in files]
    assert data["per_file"][0]["stats"]["count"] == 80


def test_stats_json_rgb(capsys, images):
    code, out, _ = run(capsys, "stats", str(images), "--format", "json", "--channels", "rgb")
    assert code == 0
    data = json.loads(out)
    assert set(data["stats"]) == {"r", "g", "b"}
    assert len(data["counts"]) == 3


def test_stats_csv(capsys, images):
    code, out, _ = run(capsys, "stats", str(images), "--format", "csv", "--per-file")
    assert code == 0
    rows = list(csv.DictReader(io.StringIO(out)))
    assert rows[0]["path"] == "*"
    assert rows[0]["channel"] == "gray"
    assert len(rows) == 4
    assert int(rows[0]["count"]) == sum(int(r["count"]) for r in rows[1:])


def test_stats_csv_rgb_per_file(capsys, images):
    _, out, _ = run(
        capsys, "stats", str(images), "--format", "csv", "--channels", "rgb", "--per-file"
    )
    rows = list(csv.DictReader(io.StringIO(out)))
    assert [r["channel"] for r in rows[:3]] == ["r", "g", "b"]
    assert len(rows) == 3 * 4


def test_stats_text_per_file(capsys, images):
    _, out, _ = run(capsys, "stats", str(images), "--per-file")
    assert str(images / "a_gray.png") in out
    _, out, _ = run(capsys, "stats", str(images), "--per-file", "--channels", "rgb")
    assert str(images / "b_rgb.PNG") in out


def test_total_equals_accumulator(capsys, images):
    _, out, _ = run(capsys, "stats", str(images), "-r", "--format", "json")
    acc = HistAccumulator()
    for p in _discover([str(images)], True, _exts("png,jpg")):
        with Image.open(p) as img:
            acc.update(img)
    assert json.loads(out)["counts"] == acc.counts.tolist()


def test_jobs_2_matches_jobs_1(capsys, tmp_path, rng):
    d = tmp_path / "many"
    d.mkdir()
    for i in range(10):
        Image.fromarray(rng.integers(0, 256, (6, 6), np.uint8)).save(d / f"{i:02}.png")
    _, serial, _ = run(capsys, "stats", str(d), "--format", "json", "--per-file", "-j", "1")
    _, parallel, _ = run(capsys, "stats", str(d), "--format", "json", "--per-file", "-j", "2")
    assert serial == parallel
    _, all_cpus, _ = run(capsys, "stats", str(d), "--format", "json", "--per-file", "-j", "0")
    assert all_cpus == serial


def test_output_file(capsys, images, tmp_path):
    out_file = tmp_path / "report.csv"
    code, out, _ = run(capsys, "stats", str(images), "--format", "csv", "-o", str(out_file))
    assert code == 0
    assert out == ""
    assert out_file.read_text().startswith("path,channel,count")


def test_bins_and_range(capsys, images):
    _, out, _ = run(
        capsys, "stats", str(images), "--format", "json", "--bins", "16", "--range", "0", "256"
    )
    data = json.loads(out)
    assert len(data["counts"]) == 16
    assert data["bins"] == 16


def test_16_bit_and_float_images(capsys, tmp_path):
    p16 = tmp_path / "deep.png"
    Image.fromarray(np.array([[0, 65535]], np.uint16)).save(p16)
    code, out, _ = run(capsys, "stats", str(p16), "--format", "json")
    assert code == 0
    assert json.loads(out)["range"] == [0.0, 65536.0]
    pf = tmp_path / "float.tif"
    Image.fromarray(np.array([[0.25, 0.75]], np.float32)).save(pf)
    code, _, err = run(capsys, "stats", str(pf))
    assert code == 1
    assert "pass --range LO HI" in err
    code, out, _ = run(capsys, "stats", str(pf), "--range", "0", "1", "--bins", "4")
    assert code == 0
    assert "max 0.75" in out


def test_mixed_bit_depths_are_skipped(capsys, tmp_path, rng):
    Image.fromarray(rng.integers(0, 256, (3, 3), np.uint8)).save(tmp_path / "a.png")
    Image.fromarray(np.array([[0, 65535]], np.uint16)).save(tmp_path / "b.png")
    code, out, err = run(capsys, "stats", str(tmp_path))
    assert code == 0
    assert "1 image (1 skipped)" in out
    assert "use --range" in err


# ---------------------------------------------------------------- errors and exit codes


def test_corrupt_file_skipped(capsys, images, corrupt):
    code, out, err = run(capsys, "stats", str(images), str(corrupt))
    assert code == 0
    assert f"imghist: skipping {corrupt}:" in err
    assert "(1 skipped)" in out
    assert "Traceback" not in err


def test_corrupt_file_strict(capsys, images, corrupt):
    code, out, err = run(capsys, "stats", str(corrupt), str(images), "--strict")
    assert code == 1
    assert out == ""
    assert "skipping" in err


def test_all_files_fail(capsys, corrupt):
    code, _, err = run(capsys, "stats", str(corrupt))
    assert code == 1
    assert "no images could be read" in err


def test_no_images_found(capsys, tmp_path):
    code, _, err = run(capsys, "stats", str(tmp_path))
    assert code == 1
    assert "no images found" in err


def test_missing_path(capsys, tmp_path, images):
    code, _, err = run(capsys, "stats", str(tmp_path / "nope"), str(images))
    assert code == 0
    assert "no such file or directory" in err


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["stats"],
        ["stats", "x", "--bins", "0"],
        ["stats", "x", "--bins", "abc"],
        ["stats", "x", "--jobs", "-1"],
        ["stats", "x", "--ext", ","],
        ["compare", "a"],
        ["compare", "a", "b", "--method", "kl"],
    ],
)
def test_usage_errors_exit_2(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("imghist ")


def test_keyboard_interrupt(capsys, images, monkeypatch):
    def boom(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("image_array_and_histogram.cli._process", boom)
    code, _, err = run(capsys, "stats", str(images))
    assert code == 130
    assert "interrupted" in err


# ---------------------------------------------------------------- compare


def test_compare_identical_files(capsys, images):
    f = str(images / "a_gray.png")
    code, out, _ = run(capsys, "compare", f, f)
    assert code == 0
    assert out == "intersection = 1.0000 (1 = identical, range [0, 1])\n"


def test_compare_directories_json(capsys, images):
    code, out, _ = run(
        capsys, "compare", str(images), str(images / "sub"), "--method", "chi2", "--format", "json"
    )
    assert code == 0
    data = json.loads(out)
    assert data["method"] == "chi2"
    assert 0 <= data["score"] <= 1
    assert data["a"]["images"] == 3
    assert data["b"]["images"] == 1


def test_compare_rgb(capsys, images):
    f = str(images / "b_rgb.PNG")
    code, out, _ = run(capsys, "compare", f, f, "--channels", "rgb", "--method", "emd")
    assert code == 0
    assert out.startswith("emd = 0.0000")


def test_compare_missing_side(capsys, images, tmp_path):
    code, _, err = run(capsys, "compare", str(images), str(tmp_path / "empty_dir_missing"))
    assert code == 1
    assert "no such file or directory" in err


def test_compare_different_bit_depths(capsys, tmp_path, rng):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    Image.fromarray(rng.integers(0, 256, (3, 3), np.uint8)).save(a)
    Image.fromarray(np.array([[0, 65535]], np.uint16)).save(b)
    code, _, err = run(capsys, "compare", str(a), str(b), "--bins", "8")
    assert code == 1
    assert "cannot compare" in err
    assert "pass --range" in err
    code, out, _ = run(capsys, "compare", str(a), str(b), "--bins", "8", "--range", "0", "65536")
    assert code == 0
    assert out.startswith("intersection = ")


def test_range_excluding_all_pixels(capsys, images):
    f = str(images / "a_gray.png")
    code, out, err = run(capsys, "stats", f, "--range", "300", "400")
    assert code == 1
    assert out == ""
    assert "no pixel values fall inside the range" in err
    code, _, err = run(capsys, "compare", f, f, "--range", "300", "400")
    assert code == 1
    assert "Traceback" not in err


def test_compare_output_file(capsys, images, tmp_path):
    f = str(images / "a_gray.png")
    out_file = tmp_path / "score.txt"
    assert main(["compare", f, f, "-o", str(out_file)]) == 0
    assert out_file.read_text().startswith("intersection = 1.0000")


# ---------------------------------------------------------------- plot


def test_plot_creates_png(capsys, images, tmp_path):
    pytest.importorskip("matplotlib")
    for channels in ("gray", "rgb"):
        out_png = tmp_path / f"hist_{channels}.png"
        code, _, _ = run(
            capsys, "stats", str(images), "--channels", channels, "--plot", str(out_png)
        )
        assert code == 0
        assert out_png.stat().st_size > 0
        with Image.open(out_png) as img:
            assert img.format == "PNG"


def test_plot_without_matplotlib(capsys, images, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    code, _, err = run(capsys, "stats", str(images), "--plot", str(tmp_path / "x.png"))
    assert code == 1
    assert "pip install 'image-array-and-histogram[plot]'" in err


# ---------------------------------------------------------------- misc


def test_sparkline():
    assert sparkline(np.array([0, 0]), 10) == "  "
    line = sparkline(np.array([0, 1, 2, 4, 8, 0, 0, 0]), 4)
    assert len(line) == 4
    assert line[-1] == " "
    assert line[2] == "█"  # columns are [1, 6, 8, 0]
    assert len(sparkline(np.arange(256), 64)) == 64


def test_python_m_help():
    result = subprocess.run(
        [sys.executable, "-m", "image_array_and_histogram", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "imghist" in result.stdout
