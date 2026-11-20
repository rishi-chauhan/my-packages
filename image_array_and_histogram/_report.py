"""Report rendering for the ``imghist`` CLI: text, JSON, CSV and plots."""

from __future__ import annotations

import csv
import io
import json
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._accumulator import HistAccumulator
from ._stats import HistStats
from ._types import CHANNEL_NAMES

BLOCKS = "▁▂▃▄▅▆▇█"
CSV_FIELDS = [
    "path", "channel", "count", "mean", "std", "min", "max", "median",
    "p1", "p5", "p25", "p75", "p95", "p99", "entropy", "clipped_low", "clipped_high",
]  # fmt: skip

StatsDict = dict[str, Any]


@dataclass
class Report:
    """Everything a stats report needs."""

    acc: HistAccumulator
    skipped: int
    integer_axis: bool  # label the right end of the axis with hi - 1 (dtype default ranges)
    per_file: list[tuple[str, StatsDict]] = field(default_factory=list)


def stats_dicts(stats: HistStats | dict[str, HistStats]) -> StatsDict:
    """Return ``to_dict()`` output, nested by channel for RGB."""
    if isinstance(stats, HistStats):
        return stats.to_dict()
    return {name: s.to_dict() for name, s in stats.items()}


def _num(v: float) -> str:
    v = float(v)
    if v.is_integer():
        return f"{int(v):,}"
    return f"{v:,.2f}" if abs(v) >= 1 else f"{v:.4g}"


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def sparkline(counts: NDArray[np.int64], width: int) -> str:
    """Render counts as one line of block characters, ``min(width, bins)`` wide."""
    cols = min(width, counts.size)
    starts = np.linspace(0, counts.size, cols + 1, dtype=np.intp)[:-1]
    col = np.add.reduceat(counts, starts)
    peak = col.max()
    if peak == 0:
        return " " * cols
    levels = np.ceil(col / peak * len(BLOCKS)).astype(np.intp)
    return "".join(BLOCKS[lvl - 1] if c else " " for c, lvl in zip(col, levels, strict=True))


def _stats_block(s: StatsDict) -> list[str]:
    pct = "  ".join(f"{k} {_num(s[k])}" for k in s if k.startswith("p") and k != "p50")
    return [
        f"count     {_num(s['count'])}",
        f"mean      {_num(s['mean']):<9} std {_num(s['std'])}",
        f"min       {_num(s['min']):<9} max {_num(s['max'])}",
        f"median    {_num(s['median']):<9} {pct}",
        f"entropy   {s['entropy']:.2f} bits",
        f"clipped   low {s['clipped_low']:.2%}  high {s['clipped_high']:.2%}",
    ]


def _axis(lo: float, hi: float, width: int) -> str:
    left, right = _num(lo), _num(hi)
    return left + " " * max(1, width - len(left) - len(right)) + right


def render_text(report: Report, width: int) -> str:
    """Render the human-readable report."""
    acc = report.acc
    assert acc.value_range is not None
    lo, hi = acc.value_range
    right = hi - 1 if report.integer_axis else hi
    stats = stats_dicts(acc.stats())
    lines = [
        f"imghist: {_plural(acc.n_images, 'image')} ({report.skipped} skipped) · "
        f"channel: {acc.channels} · {acc.bins} bins"
    ]
    blocks = [("", stats, acc.counts)] if acc.channels == "gray" else [
        (name.upper(), stats[name], acc.counts[i]) for i, name in enumerate(CHANNEL_NAMES)
    ]  # fmt: skip
    for title, s, counts in blocks:
        lines.append("")
        if title:
            lines.append(title)
        lines.extend(_stats_block(s))
        line = sparkline(counts, width)
        lines += ["", line, _axis(lo, right, len(line))]
    for path, s in report.per_file:
        lines += ["", path]
        for name, sub in [("", s)] if "count" in s else [(k.upper(), s[k]) for k in s]:
            if name:
                lines.append(name)
            lines.extend("  " + ln for ln in _stats_block(sub))
    return "\n".join(lines) + "\n"


def render_json(report: Report, version: str) -> str:
    """Render the machine-readable JSON report."""
    acc = report.acc
    assert acc.value_range is not None
    out: dict[str, Any] = {
        "version": version,
        "images": acc.n_images,
        "skipped": report.skipped,
        "channels": acc.channels,
        "bins": acc.bins,
        "range": [float(v) for v in acc.value_range],
        "stats": stats_dicts(acc.stats()),
        "counts": acc.counts.tolist(),
    }
    if report.per_file:
        out["per_file"] = [{"path": p, "stats": s} for p, s in report.per_file]
    return json.dumps(out, indent=2) + "\n"


def _csv_rows(path: str, s: StatsDict) -> list[dict[str, Any]]:
    channels = [("gray", s)] if "count" in s else list(s.items())
    return [
        {"path": path, "channel": ch, **{k: d.get(k) for k in CSV_FIELDS[2:]}} for ch, d in channels
    ]


def render_csv(report: Report) -> str:
    """Render one CSV row per channel for the total (path ``*``) and each file."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(_csv_rows("*", stats_dicts(report.acc.stats())))
    for path, s in report.per_file:
        writer.writerows(_csv_rows(path, s))
    return buf.getvalue()


def save_plot(acc: HistAccumulator, path: str) -> int:
    """Save a histogram plot; return an exit code (1 if matplotlib is missing)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "imghist: --plot requires matplotlib: pip install 'image-array-and-histogram[plot]'",
            file=sys.stderr,
        )
        return 1
    edges = acc.edges
    fig, ax = plt.subplots(figsize=(8, 4))
    if acc.channels == "gray":
        ax.stairs(acc.counts, edges, fill=True, color="0.3")
    else:
        for row, color in zip(acc.counts, ("red", "green", "blue"), strict=True):
            ax.stairs(row, edges, fill=True, color=color, alpha=0.5, label=color)
        ax.legend()
    ax.set_xlabel("pixel value")
    ax.set_ylabel("count")
    ax.set_title(f"imghist: {_plural(acc.n_images, 'image')}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return 0
