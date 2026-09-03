#!/usr/bin/env python3
"""
Generate a portfolio-friendly README demo SVG from the included synthetic
fireworks scenario data.

The output is intentionally a designed overview rather than a raw matplotlib
export, but every displayed metric is derived from the checked-in scenario
metadata and positions files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from xml.sax.saxutils import escape


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = REPO_ROOT / "test_data" / "scenario3_fireworks"
POSITIONS_PATH = SCENARIO_DIR / "positions.json"
METADATA_PATH = SCENARIO_DIR / "metadata.json"
OUTPUT_SVG = REPO_ROOT / "docs" / "demo_diagram.svg"

WIDTH = 1280
HEIGHT = 760


def load_inputs() -> tuple[dict, dict]:
    with POSITIONS_PATH.open() as f:
        positions = json.load(f)
    with METADATA_PATH.open() as f:
        metadata = json.load(f)
    return positions, metadata


def scale_points(points: list[list[float]], event_xy: list[float]) -> tuple[list[tuple[float, float]], tuple[float, float]]:
    all_x = [p[0] for p in points] + [event_xy[0]]
    all_y = [p[1] for p in points] + [event_xy[1]]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    panel_x = 82
    panel_y = 126
    panel_w = 664
    panel_h = 548
    pad = 66

    usable_w = panel_w - 2 * pad
    usable_h = panel_h - 2 * pad

    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    scale = min(usable_w / span_x, usable_h / span_y)

    def map_point(x: float, y: float) -> tuple[float, float]:
        sx = panel_x + pad + (x - min_x) * scale
        sy = panel_y + panel_h - pad - (y - min_y) * scale
        return round(sx, 2), round(sy, 2)

    return [map_point(x, y) for x, y in points], map_point(event_xy[0], event_xy[1])


def max_baseline(points: list[list[float]]) -> float:
    best = 0.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            best = max(best, math.hypot(dx, dy))
    return best


def rect(x: int, y: int, w: int, h: int, rx: int, fill: str, stroke: str | None = None, opacity: float | None = None) -> str:
    attrs = [f'x="{x}"', f'y="{y}"', f'width="{w}"', f'height="{h}"', f'rx="{rx}"', f'fill="{fill}"']
    if stroke:
        attrs.append(f'stroke="{stroke}"')
    if opacity is not None:
        attrs.append(f'fill-opacity="{opacity}"')
    return f"<rect {' '.join(attrs)} />"


def text(x: float, y: float, value: str, size: int, fill: str, weight: int = 400, family: str = "IBM Plex Sans") -> str:
    return (
        f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" '
        f'font-family="{family}, Segoe UI, sans-serif" font-weight="{weight}">{escape(value)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float, dash: str | None = None, opacity: float | None = None) -> str:
    attrs = [
        f'x1="{x1}"',
        f'y1="{y1}"',
        f'x2="{x2}"',
        f'y2="{y2}"',
        f'stroke="{stroke}"',
        f'stroke-width="{width}"',
        'stroke-linecap="round"',
    ]
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    if opacity is not None:
        attrs.append(f'stroke-opacity="{opacity}"')
    return f"<line {' '.join(attrs)} />"


def circle(cx: float, cy: float, r: float, fill: str, stroke: str | None = None, stroke_width: float | None = None, opacity: float | None = None) -> str:
    attrs = [f'cx="{cx}"', f'cy="{cy}"', f'r="{r}"', f'fill="{fill}"']
    if stroke:
        attrs.append(f'stroke="{stroke}"')
    if stroke_width is not None:
        attrs.append(f'stroke-width="{stroke_width}"')
    if opacity is not None:
        attrs.append(f'fill-opacity="{opacity}"')
    return f"<circle {' '.join(attrs)} />"


def build_metric_card(x: int, y: int, title: str, value: str, accent: str, value_size: int = 28) -> str:
    return "\n".join(
        [
            rect(x, y, 186, 104, 22, "#F8FBFF", "#D5E2F0"),
            rect(x + 18, y + 18, 38, 8, 4, accent),
            text(x + 18, y + 48, title, 15, "#64748B", 500),
            text(x + 18, y + 82, value, value_size, "#0F172A", 700),
        ]
    )


def build_arrival_rows(files: list[str], arrivals_s: list[float], x: int, y: int, width: int) -> str:
    rows = []
    min_arrival = min(arrivals_s)
    max_delta_ms = max((arrival - min_arrival) * 1000.0 for arrival in arrivals_s) or 1.0
    sorted_rows = sorted(zip(files, arrivals_s), key=lambda item: item[1])

    for idx, (file_name, arrival) in enumerate(sorted_rows):
        row_y = y + idx * 30
        delta_ms = (arrival - min_arrival) * 1000.0
        bar_w = 110 + int((delta_ms / max_delta_ms) * 160) if max_delta_ms else 110
        rows.append(rect(x, row_y - 14, width, 26, 13, "#F8FBFF", "#DCE7F1"))
        rows.append(text(x + 14, row_y + 3, file_name.replace(".mp4", ""), 14, "#0F172A", 600))
        rows.append(rect(x + 104, row_y - 6, bar_w, 10, 5, "#D9EAFE"))
        rows.append(rect(x + 104, row_y - 6, 84, 10, 5, "#2563EB"))
        rows.append(text(x + width - 92, row_y + 3, f"+{delta_ms:0.1f} ms", 13, "#1E293B", 600))
    return "\n".join(rows)


def build_svg(positions: dict, metadata: dict) -> str:
    mic_positions = metadata["microphone_positions_m"]
    event_xy = metadata["source_position_m"]
    screen_points, event_point = scale_points(mic_positions, event_xy)
    mics = positions["mics"]
    event_info = positions["event"]

    files = [mic["file"] for mic in mics]
    arrivals = metadata["arrival_times_s"]
    clock_offsets = metadata["clock_offsets_s"]
    pair_count = len(files) * (len(files) - 1) // 2
    sample_rate_khz = metadata["sample_rate_hz"] / 1000.0
    baseline_m = max_baseline(mic_positions)
    min_offset_ms = min(clock_offsets) * 1000.0
    max_offset_ms = max(clock_offsets) * 1000.0

    parts = [
        f'<svg width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" fill="none" xmlns="http://www.w3.org/2000/svg">',
        """
<defs>
  <linearGradient id="bg" x1="48" y1="36" x2="1216" y2="712" gradientUnits="userSpaceOnUse">
    <stop stop-color="#06131F"/>
    <stop offset="1" stop-color="#132B45"/>
  </linearGradient>
  <linearGradient id="panel" x1="94" y1="108" x2="1108" y2="682" gradientUnits="userSpaceOnUse">
    <stop stop-color="#F8FBFF" stop-opacity="0.98"/>
    <stop offset="1" stop-color="#EDF4FB" stop-opacity="0.98"/>
  </linearGradient>
  <linearGradient id="halo" x1="0" y1="0" x2="1" y2="1">
    <stop stop-color="#FFB703"/>
    <stop offset="1" stop-color="#FB7185"/>
  </linearGradient>
  <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
    <feDropShadow dx="0" dy="18" stdDeviation="24" flood-color="#020617" flood-opacity="0.24"/>
  </filter>
</defs>
        """.strip(),
        rect(18, 18, WIDTH - 36, HEIGHT - 36, 30, "url(#bg)"),
        circle(170, 128, 160, "#0EA5E9", None, None, 0.10),
        circle(1100, 94, 150, "#FB7185", None, None, 0.10),
        rect(42, 42, WIDTH - 84, HEIGHT - 84, 30, "url(#panel)", None, 0.96),
        rect(58, 58, 712, 628, 28, "#0F172A", "#1E293B"),
        rect(792, 58, 430, 628, 28, "#F3F8FD", "#D5E2F0"),
        text(82, 94, "Portfolio Demo", 16, "#94A3B8", 700),
        text(82, 132, "Acoustic Event Locator", 38, "#F8FAFC", 700),
        text(82, 164, positions["description"], 18, "#CBD5E1", 500),
        text(82, 196, "Generated from the included synthetic fireworks scenario data", 16, "#93C5FD", 500),
        text(816, 98, "Scenario Snapshot", 16, "#64748B", 700),
        text(816, 132, (event_info.get("true_location") or event_info["estimated_location"])["description"], 28, "#0F172A", 700),
        text(816, 164, f'Event type: {event_info["type"]}  |  UTC: {event_info["time_utc"]}', 15, "#475569", 500),
    ]

    parts.extend(
        [
            build_metric_card(816, 196, "Cameras", str(len(files)), "#2563EB"),
            build_metric_card(1018, 196, "TDOA Pairs", str(pair_count), "#0F766E"),
            build_metric_card(816, 320, "Sample Rate", f"{sample_rate_khz:.0f} kHz", "#EA580C"),
            build_metric_card(1018, 320, "Clock Skew", f"{min_offset_ms:.1f} to +{max_offset_ms:.1f} ms", "#7C3AED", 21),
        ]
    )

    parts.extend(
        [
            text(816, 452, "First Arrival Order", 19, "#0F172A", 700),
            text(816, 474, f"Across a {baseline_m:.1f} m aperture, relative to earliest hit", 14, "#64748B", 500),
            build_arrival_rows(files, arrivals, 816, 514, 366),
        ]
    )

    grid_left = 106
    grid_top = 150
    grid_right = 722
    grid_bottom = 650
    for gx in range(grid_left, grid_right + 1, 74):
        parts.append(line(gx, grid_top, gx, grid_bottom, "#334155", 1, None, 0.32))
    for gy in range(grid_top, grid_bottom + 1, 74):
        parts.append(line(grid_left, gy, grid_right, gy, "#334155", 1, None, 0.32))

    parts.append(rect(84, 224, 170, 88, 20, "#111C2B", "#20324A"))
    parts.append(text(102, 254, "Geometry View", 20, "#F8FAFC", 700))
    parts.append(text(102, 280, "L-shaped camera array", 15, "#93C5FD", 500))
    parts.append(text(102, 302, "Real coordinates from test_data/", 13, "#94A3B8", 500))

    ex, ey = event_point
    parts.append(circle(ex, ey, 76, "#F59E0B", "#FCD34D", 2, 0.10))
    parts.append(circle(ex, ey, 126, "#FB7185", "#FDA4AF", 2, 0.06))

    for idx, ((sx, sy), mic) in enumerate(zip(screen_points, mics), start=1):
        parts.append(line(ex, ey, sx, sy, "#F8FAFC", 1.5, "6 8", 0.45))
        parts.append(circle(sx, sy, 13, "#0EA5E9", "#E0F2FE", 3))
        label_w = 112 if len(mic["description"]) < 18 else 152
        label_x = sx + 18 if sx < 520 else sx - label_w - 18
        label_y = sy - 34 if sy > 220 else sy + 18
        parts.append(rect(int(label_x), int(label_y), label_w, 46, 16, "#E6F4FF", "#C7D9E8"))
        parts.append(text(label_x + 12, label_y + 20, f"cam{idx}", 14, "#0F172A", 700))
        parts.append(text(label_x + 12, label_y + 37, mic["description"], 10, "#475569", 500))

    event_star = [
        (ex, ey - 18),
        (ex + 6, ey - 6),
        (ex + 18, ey),
        (ex + 6, ey + 6),
        (ex, ey + 18),
        (ex - 6, ey + 6),
        (ex - 18, ey),
        (ex - 6, ey - 6),
    ]
    star_points = " ".join(f"{x},{y}" for x, y in event_star)
    parts.append(f'<polygon points="{star_points}" fill="url(#halo)" stroke="#FFF7ED" stroke-width="3" />')
    parts.append(rect(int(ex + 22), int(ey - 26), 170, 58, 18, "#FFF7ED", "#FED7AA"))
    parts.append(text(ex + 38, ey - 2, "synthetic source", 15, "#7C2D12", 700))
    parts.append(text(ex + 38, ey + 20, "derived from metadata.json", 12, "#9A3412", 500))

    parts.extend(
        [
            text(88, 708, "This asset is script-generated for the public README.", 15, "#CBD5E1", 500),
            text(88, 730, "Regenerate with: python docs/generate_demo_diagram.py", 15, "#F8FAFC", 600, "IBM Plex Mono"),
        ]
    )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    positions, metadata = load_inputs()
    svg = build_svg(positions, metadata)
    OUTPUT_SVG.write_text(svg)
    print(f"Wrote {OUTPUT_SVG}")


if __name__ == "__main__":
    main()
