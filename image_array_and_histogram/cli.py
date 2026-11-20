"""The ``imghist`` command-line tool: histograms and statistics for images and folders."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from . import __version__
from ._accumulator import HistAccumulator
from ._compare import METHOD_INFO, compare_hist
from ._report import Report, StatsDict, render_csv, render_json, render_text, save_plot, stats_dicts
from ._types import Channels

DEFAULT_EXTS = ".png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp,.gif"
_EXPECTED_ERRORS = (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError)


class _StrictFailure(Exception):
    """Raised to stop processing on the first unreadable file with ``--strict``."""


@dataclass(frozen=True)
class _Job:
    channels: Channels
    bins: int
    value_range: tuple[float, float] | None
    per_file: bool


@dataclass(frozen=True)
class _Result:
    path: str
    acc: HistAccumulator | None = None
    stats: StatsDict | None = None
    error: str | None = None


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


def _non_negative_int(text: str) -> int:
    return 0 if text.strip() == "0" else _positive_int(text)


def _exts(text: str) -> frozenset[str]:
    exts = {e.strip().lower() for e in text.split(",") if e.strip()}
    if not exts:
        raise argparse.ArgumentTypeError("expected at least one extension")
    return frozenset(e if e.startswith(".") else f".{e}" for e in exts)


def _discover(paths: Sequence[str], recursive: bool, exts: frozenset[str]) -> list[Path]:
    """Expand files and directories into a sorted, de-duplicated list of image files.

    Explicitly named files are always included; inside directories only files
    with a matching extension are kept and hidden files/directories are skipped.
    """
    found: dict[Path, Path] = {}
    for raw in paths:
        root = Path(raw)
        if root.is_file():
            candidates: Iterator[Path] = iter([root])
        elif root.is_dir():
            walk = root.rglob("*") if recursive else root.iterdir()
            candidates = (
                p
                for p in walk
                if p.suffix.lower() in exts
                and not any(part.startswith(".") for part in p.relative_to(root).parts)
                and p.is_file()
            )
        else:
            print(f"imghist: skipping {raw}: no such file or directory", file=sys.stderr)
            continue
        for p in candidates:
            found.setdefault(p.resolve(), p)
    return sorted(found.values())


def _process(path: Path, job: _Job) -> _Result:
    """Histogram one file (top-level so it can run in a worker process)."""
    try:
        acc = HistAccumulator(channels=job.channels, bins=job.bins, value_range=job.value_range)
        with Image.open(path) as img:
            img.load()
            acc.update(img)
        stats = stats_dicts(acc.stats()) if job.per_file else None
    except _EXPECTED_ERRORS as exc:
        return _Result(str(path), error=str(exc) or type(exc).__name__)
    return _Result(str(path), acc=acc, stats=stats)


def _run(paths: list[Path], job: _Job, jobs: int) -> Iterator[_Result]:
    workers = jobs or os.cpu_count() or 1
    if workers < 2 or len(paths) < 2 * workers:
        yield from (_process(p, job) for p in paths)
        return
    executor = ProcessPoolExecutor(max_workers=workers)
    try:
        chunk = max(1, len(paths) // (workers * 4))
        yield from executor.map(_process, paths, [job] * len(paths), chunksize=chunk)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _accumulate(paths: list[Path], job: _Job, args: argparse.Namespace) -> tuple[Report, int]:
    """Process files into a report; return it with the number of failures."""
    total = HistAccumulator(channels=job.channels, bins=job.bins, value_range=job.value_range)
    report = Report(total, skipped=0, integer_axis=job.value_range is None)
    for res in _run(paths, job, args.jobs):
        error = res.error
        if res.acc is not None:
            try:
                total.merge(res.acc)
            except ValueError as exc:
                error = f"{exc} (use --range to mix bit depths)"
        if error is not None:
            if "explicit value_range" in error:
                error += " (pass --range LO HI)"
            print(f"imghist: skipping {res.path}: {error}", file=sys.stderr)
            report.skipped += 1
            if args.strict:
                raise _StrictFailure
        elif res.stats is not None:
            report.per_file.append((res.path, res.stats))
    return report, report.skipped


def _job(args: argparse.Namespace, per_file: bool = False) -> _Job:
    return _Job(args.channels, args.bins, tuple(args.range) if args.range else None, per_file)


def _collect(paths: Sequence[str], args: argparse.Namespace, job: _Job) -> Report | None:
    files = _discover(paths, args.recursive, args.ext)
    if not files:
        print(f"imghist: no images found in {' '.join(paths)}", file=sys.stderr)
        return None
    report, _ = _accumulate(files, job, args)
    if report.acc.n_images == 0:
        print("imghist: no images could be read", file=sys.stderr)
        return None
    if not report.acc.counts.any():
        print(f"imghist: no pixel values fall inside the range {job.value_range}", file=sys.stderr)
        return None
    return report


def _write(text: str, output: str | None) -> None:
    if output is None:
        sys.stdout.write(text)
    else:
        Path(output).write_text(text, encoding="utf-8", newline="")


def _cmd_stats(args: argparse.Namespace) -> int:
    report = _collect(args.paths, args, _job(args, args.per_file))
    if report is None:
        return 1
    if args.format == "json":
        text = render_json(report, __version__)
    elif args.format == "csv":
        text = render_csv(report)
    else:
        text = render_text(report, args.width)
    _write(text, args.output)
    return save_plot(report.acc, args.plot) if args.plot else 0


def _cmd_compare(args: argparse.Namespace) -> int:
    job = _job(args)
    a, b = _collect([args.a], args, job), _collect([args.b], args, job)
    if a is None or b is None:
        return 1
    if a.acc.value_range != b.acc.value_range:
        print(
            f"imghist: cannot compare: A has range {a.acc.value_range} but B has "
            f"{b.acc.value_range} (different bit depths); pass --range LO HI",
            file=sys.stderr,
        )
        return 1
    score = compare_hist(a.acc.counts, b.acc.counts, method=args.method)
    if args.format == "json":
        side = {
            name: {
                "path": path,
                "images": r.acc.n_images,
                "skipped": r.skipped,
                "stats": stats_dicts(r.acc.stats()),
            }
            for name, path, r in (("a", args.a, a), ("b", args.b, b))
        }
        import json

        text = json.dumps({"method": args.method, "score": score, **side}, indent=2) + "\n"
    else:
        rng, ident = METHOD_INFO[args.method]
        text = f"{args.method} = {score:.4f} ({ident:g} = identical, range {rng})\n"
    _write(text, args.output)
    return 0


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-r", "--recursive", action="store_true", help="recurse into directories")
    p.add_argument(
        "--ext",
        type=_exts,
        default=_exts(DEFAULT_EXTS),
        help=f"comma-separated extensions to include (default: {DEFAULT_EXTS})",
    )
    p.add_argument("--channels", choices=("gray", "rgb"), default="gray", help="default: gray")
    p.add_argument("--bins", type=_positive_int, default=256, help="number of bins (default: 256)")
    p.add_argument(
        "--range",
        nargs=2,
        type=float,
        metavar=("LO", "HI"),
        help="histogram range; required for float images",
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=_non_negative_int,
        default=1,
        help="worker processes; 0 = one per CPU (default: 1)",
    )
    p.add_argument("--strict", action="store_true", help="fail on the first unreadable file")
    p.add_argument("-o", "--output", metavar="FILE", help="write the report to FILE")


def build_parser() -> argparse.ArgumentParser:
    """Build the ``imghist`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="imghist", description="Histograms and exposure statistics for images and folders."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    stats = sub.add_parser("stats", help="histogram and statistics of images or folders")
    stats.add_argument("paths", nargs="+", metavar="PATH", help="image files or directories")
    _add_common(stats)
    stats.add_argument("--per-file", action="store_true", help="also report each file")
    stats.add_argument("--format", choices=("text", "json", "csv"), default="text")
    stats.add_argument(
        "--width",
        type=_positive_int,
        default=64,
        help="text histogram width in columns (default: 64)",
    )
    stats.add_argument("--plot", metavar="FILE", help="save a plot (needs the [plot] extra)")
    stats.set_defaults(func=_cmd_stats)

    cmp = sub.add_parser("compare", help="compare the histograms of two images or folders")
    cmp.add_argument("a", metavar="A", help="image file or directory")
    cmp.add_argument("b", metavar="B", help="image file or directory")
    cmp.add_argument("--method", choices=list(METHOD_INFO), default="intersection")
    cmp.add_argument("--format", choices=("text", "json"), default="text")
    _add_common(cmp)
    cmp.set_defaults(func=_cmd_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``imghist`` and return its exit code.

    Exit codes: 0 success; 1 no images found, all images failed, or a
    ``--strict`` failure; 2 usage error; 130 interrupted.
    """
    args = build_parser().parse_args(argv)
    try:
        code: int = args.func(args)
    except _StrictFailure:
        return 1
    except KeyboardInterrupt:
        print("imghist: interrupted", file=sys.stderr)
        return 130
    return code
