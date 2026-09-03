#!/usr/bin/env python3
"""
Generate the README hero diagram (docs/demo_diagram.svg) from an actual locator run.

  python docs/generate_demo_diagram.py                 # scenario3_fireworks, seed 0
  python docs/generate_demo_diagram.py --scenario scenario4_window_shot --out /tmp/demo.svg

The script synthesizes the scenario's recordings with generate_test_data (deterministic seed),
runs locate_event.locate_from_signals on them exactly as the command line would, and draws
what came out: the estimate with its 95% ellipse (to scale, plus a zoomed inset), the solved
height against the truth, and every recording's arrival, onset SNR and residual. Nothing in
the picture is typed in by hand.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import tempfile
import textwrap
from xml.sax.saxutils import escape

import numpy as np
import soundfile as sf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import generate_test_data as gen  # noqa: E402
import locate_event as le  # noqa: E402

WIDTH, HEIGHT = 1280, 760
FONT = "IBM Plex Sans, Inter, Segoe UI, Helvetica Neue, Arial, sans-serif"
MONO = "IBM Plex Mono, JetBrains Mono, Menlo, Consolas, monospace"
C_CAM, C_EST, C_TRUE, C_TEXT, C_MUTED = "#38BDF8", "#F59E0B", "#22C55E", "#0F172A", "#64748B"


# ------------------------------ run the locator ------------------------------


def run_locator(scenario: str, seed: int) -> dict:
    sdir = os.path.join(REPO_ROOT, "test_data", scenario)
    tmp = tempfile.mkdtemp()
    try:
        d = os.path.join(tmp, scenario)
        os.makedirs(d)
        shutil.copy(os.path.join(sdir, "positions.json"), d)
        truth = gen.generate_scenario(d, fmt="wav", seed=seed)
        mics, (lat0, lon0), c, J = le.load_positions(os.path.join(d, "positions.json"), d)
        XYZ = le.mic_local_xyz(mics, lat0, lon0)
        hsig = le.mic_height_sigma(mics)
        tracks = [sf.read(m.file, dtype="float64")[0] for m in mics]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    hp = truth.get("height_prior_m")
    p = (le.PipelineParams(source_z=hp["mean"], source_z_sigma=hp["sigma"]) if hp
         else le.PipelineParams(source_z=truth["source_height_m"]))
    res = le.locate_from_signals(tracks, truth["sample_rate_hz"], XYZ, c, p, height_sigma=hsig)
    return dict(scenario=scenario, positions=J, truth=truth, XYZ=XYZ, hsig=hsig, c=c, params=p, res=res)


# ------------------------------ SVG primitives ------------------------------


def rect(x, y, w, h, rx, fill, stroke=None, opacity=None, sw=1):
    a = [f'x="{x:.1f}"', f'y="{y:.1f}"', f'width="{w:.1f}"', f'height="{h:.1f}"', f'rx="{rx}"', f'fill="{fill}"']
    if stroke:
        a += [f'stroke="{stroke}"', f'stroke-width="{sw}"']
    if opacity is not None:
        a.append(f'fill-opacity="{opacity}"')
    return f"<rect {' '.join(a)} />"


def text(x, y, value, size, fill, weight=400, anchor="start", family=FONT, opacity=None):
    a = [f'x="{x:.1f}"', f'y="{y:.1f}"', f'fill="{fill}"', f'font-size="{size}"', f'font-family="{family}"',
         f'font-weight="{weight}"', f'text-anchor="{anchor}"']
    if opacity is not None:
        a.append(f'fill-opacity="{opacity}"')
    return f"<text {' '.join(a)}>{escape(str(value))}</text>"


def line(x1, y1, x2, y2, stroke, width=1.0, dash=None, opacity=None):
    a = [f'x1="{x1:.1f}"', f'y1="{y1:.1f}"', f'x2="{x2:.1f}"', f'y2="{y2:.1f}"', f'stroke="{stroke}"',
         f'stroke-width="{width}"', 'stroke-linecap="round"']
    if dash:
        a.append(f'stroke-dasharray="{dash}"')
    if opacity is not None:
        a.append(f'stroke-opacity="{opacity}"')
    return f"<line {' '.join(a)} />"


def circle(cx, cy, r, fill, stroke=None, sw=None, opacity=None):
    a = [f'cx="{cx:.1f}"', f'cy="{cy:.1f}"', f'r="{r}"', f'fill="{fill}"']
    if stroke:
        a.append(f'stroke="{stroke}"')
    if sw is not None:
        a.append(f'stroke-width="{sw}"')
    if opacity is not None:
        a.append(f'fill-opacity="{opacity}"')
    return f"<circle {' '.join(a)} />"


def star(cx, cy, r, fill, stroke="#FFF7ED", sw=2.0):
    pts = []
    for k in range(10):
        ang = -math.pi / 2 + k * math.pi / 5
        rr = r if k % 2 == 0 else 0.45 * r
        pts.append(f"{cx + rr * math.cos(ang):.1f},{cy + rr * math.sin(ang):.1f}")
    return f'<polygon points="{" ".join(pts)}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round" />'


def triangle(cx, cy, r, fill, stroke="#E0F2FE", sw=2.0):
    pts = [(cx, cy - r), (cx + 0.87 * r, cy + 0.5 * r), (cx - 0.87 * r, cy + 0.5 * r)]
    return f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round" />'


def ellipse(cx, cy, rx, ry, angle_deg, stroke, sw=2.0, fill="none", opacity=None):
    a = [f'cx="{cx:.1f}"', f'cy="{cy:.1f}"', f'rx="{max(rx, 0.5):.2f}"', f'ry="{max(ry, 0.5):.2f}"', f'fill="{fill}"',
         f'stroke="{stroke}"', f'stroke-width="{sw}"', f'transform="rotate({-angle_deg:.2f} {cx:.1f} {cy:.1f})"']
    if opacity is not None:
        a.append(f'fill-opacity="{opacity}"')
    return f"<ellipse {' '.join(a)} />"


def fit_text(s: str, max_chars: int) -> str:
    return s if len(s) <= max_chars else s[: max_chars - 1].rstrip() + "…"


# ------------------------------ panels ------------------------------


class Frame:
    """Maps metres to pixels with equal aspect inside a pixel box."""

    def __init__(self, pts_xy, box, pad_m=6.0):
        pts = np.asarray(pts_xy, float)
        lo, hi = pts.min(0) - pad_m, pts.max(0) + pad_m
        x0, y0, w, h = box
        self.scale = min(w / (hi[0] - lo[0]), h / (hi[1] - lo[1]))
        cx, cy = 0.5 * (lo + hi)
        self.ox = x0 + w / 2 - cx * self.scale
        self.oy = y0 + h / 2 + cy * self.scale
        self.box = box

    def px(self, x, y):
        return self.ox + x * self.scale, self.oy - y * self.scale

    def nice_ticks(self, lo, hi, target=6):
        span = hi - lo
        step = 10 ** math.floor(math.log10(span / target))
        for m in (1, 2, 5, 10):
            if span / (step * m) <= target:
                step *= m
                break
        start = math.ceil(lo / step) * step
        return [start + k * step for k in range(int((hi - start) / step) + 1)], step


def draw_map(parts, S, box):
    truth, XYZ, res = S["truth"], S["XYZ"], S["res"]
    sol = res["solution"]
    src = np.array(truth["source_position_m"])
    fr = Frame(np.vstack([XYZ[:, :2], src[None, :]]), box, pad_m=7.0)
    x0, y0, w, h = box
    # axes grid in metres
    inv = lambda px, py: ((px - fr.ox) / fr.scale, (fr.oy - py) / fr.scale)  # noqa: E731
    xlo, yhi = inv(x0, y0)
    xhi, ylo = inv(x0 + w, y0 + h)
    xt, step = fr.nice_ticks(xlo, xhi)
    yt, _ = fr.nice_ticks(ylo, yhi)
    for gx in xt:
        px, _ = fr.px(gx, 0)
        parts.append(line(px, y0, px, y0 + h, "#334155", 1, None, 0.55))
        parts.append(text(px, y0 + h + 16, f"{gx:g}", 11, "#94A3B8", 500, "middle"))
    for gy in yt:
        _, py = fr.px(0, gy)
        parts.append(line(x0, py, x0 + w, py, "#334155", 1, None, 0.55))
        parts.append(text(x0 - 8, py + 4, f"{gy:g}", 11, "#94A3B8", 500, "end"))
    parts.append(text(x0 + w / 2, y0 + h + 34, "x east (m)", 12, "#94A3B8", 600, "middle"))
    parts.append(text(x0 - 36, y0 + h / 2, "y north (m)", 12, "#94A3B8", 600, "middle") .replace("<text ", f'<text transform="rotate(-90 {x0 - 36:.1f} {y0 + h / 2:.1f})" ', 1))

    ex, ey = fr.px(*sol.s_xy)
    tx, ty = fr.px(*src)
    # rays from the estimate to each camera, arrival order labels
    order = np.argsort([tr.arrival_s if tr.arrival_s is not None else np.inf for tr in res["tracks"]])
    rank = {int(i): k + 1 for k, i in enumerate(order)}
    for i, (mx, my, _) in enumerate(XYZ):
        px, py = fr.px(mx, my)
        parts.append(line(ex, ey, px, py, "#F8FAFC", 1.2, "5 7", 0.35))
    # cameras with labels placed away from the array's arms
    names = [os.path.splitext(f)[0] for f in truth["files"]]
    cx_all = XYZ[:, 0].mean()
    cy_all = XYZ[:, 1].mean()
    for i, (mx, my, mz) in enumerate(XYZ):
        px, py = fr.px(mx, my)
        used = res["tracks"][i].used
        parts.append(triangle(px, py, 10, C_CAM if used else "#64748B"))
        # label to the outside of the array relative to its centroid
        dx = -1 if mx < cx_all - 1e-6 else (1 if mx > cx_all + 1e-6 else 0)
        dy = -1 if my < cy_all - 1e-6 else (1 if my > cy_all + 1e-6 else 0)
        lx = px + (16 if dx > 0 else (-16 if dx < 0 else 0))
        ly = py + (26 if dy < 0 else (-16 if dy > 0 else 5))
        anchor = "start" if dx > 0 else ("end" if dx < 0 else "middle")
        if dx == 0 and dy == 0:
            lx, ly, anchor = px + 16, py + 5, "start"
        label = f"{names[i]}  #{rank.get(i, '-')}  {mz:g} m"
        parts.append(text(lx, ly, label, 11, "#E2E8F0", 600, anchor))
    # truth, estimate, ellipse
    a, b, ang = le.ellipse_from_cov2(sol.cov_xy)
    parts.append(ellipse(ex, ey, a * fr.scale, b * fr.scale, ang, C_EST, 2.0, C_EST, 0.18))
    for alt in sol.alternatives:  # only alternatives that fit about as well (the ambiguous ones)
        if alt["delta_cost"] <= 3.0:
            ax_, ay_ = fr.px(alt["x"], alt["y"])
            parts.append(circle(ax_, ay_, 9, "none", "#FB7185", 2.5))
    parts.append(star(ex, ey, 12, C_EST, "#FFF7ED", 2))
    parts.append(star(tx, ty, 8, C_TRUE, "#DCFCE7", 1.5))

    # zoom inset: top-right corner of the map box unless the estimate is there
    iw, ih = 236, 176
    ix = x0 + w - iw - 6
    iy = y0 + 6
    if ex > ix - 30 and ey < iy + ih + 30:
        ix = x0 + 6
    parts.append(rect(ix, iy, iw, ih, 16, "#0B1526", "#38BDF8", 0.96, 1.5))
    zs = 0.32 * min(iw, ih) / max(a, 0.05)  # px per meter so the ellipse fills ~ the inset
    zs = min(zs, 220.0)
    zcx, zcy = ix + iw / 2, iy + ih / 2 + 16
    zx = lambda X, Y: (zcx + (X - sol.s_xy[0]) * zs, zcy - (Y - sol.s_xy[1]) * zs)  # noqa: E731
    parts.append(ellipse(zcx, zcy, a * zs, b * zs, ang, C_EST, 2.0, C_EST, 0.15))
    txz, tyz = zx(*src)
    parts.append(star(zcx, zcy, 12, C_EST, "#FFF7ED", 2))
    if ix <= txz <= ix + iw and iy <= tyz <= iy + ih:
        parts.append(star(txz, tyz, 7, C_TRUE, "#DCFCE7", 1.5))
    bar_m = 1.0 if zs * 1.0 < iw * 0.6 else 0.5
    bx1, by = ix + 16, iy + ih - 16
    parts.append(line(bx1, by, bx1 + bar_m * zs, by, "#E2E8F0", 2))
    parts.append(text(bx1 + bar_m * zs / 2, by - 6, f"{bar_m:g} m", 10, "#E2E8F0", 600, "middle"))
    err = float(np.linalg.norm(sol.s_xy - src))
    parts.append(text(ix + 14, iy + 20, "Zoom on the estimate", 12, "#E2E8F0", 700))
    parts.append(text(ix + 14, iy + 36, f"95% ellipse {2*a:.2f} × {2*b:.2f} m, error {err*100:.1f} cm", 10.5, "#93C5FD", 500))
    parts.append(line(ix + (0 if ex < ix else iw), iy + ih / 2, ex, ey, "#38BDF8", 1, "3 5", 0.6))

    # legend row under the axis label
    lx, ly = x0, y0 + h + 60
    parts.append(triangle(lx + 8, ly - 4, 7, C_CAM))
    parts.append(text(lx + 22, ly, "camera: name, #arrival order, height", 11, "#E2E8F0", 500))
    parts.append(star(lx + 262, ly - 4, 8, C_TRUE, "#DCFCE7", 1.5))
    parts.append(text(lx + 276, ly, "true source", 11, "#E2E8F0", 500))
    parts.append(star(lx + 372, ly - 4, 8, C_EST, "#FFF7ED", 1.5))
    parts.append(text(lx + 386, ly, "estimate with 95% ellipse", 11, "#E2E8F0", 500))
    if sol.alternatives:
        parts.append(circle(lx + 560, ly - 4, 6, "none", "#FB7185", 2))
        parts.append(text(lx + 572, ly, "other local minimum", 11, "#E2E8F0", 500))
    return fr


def metric_card(parts, x, y, w, title, value, accent, sub=None, value_size=24):
    parts.append(rect(x, y, w, 88, 18, "#F8FBFF", "#D5E2F0"))
    parts.append(rect(x + 16, y + 14, 34, 6, 3, accent))
    parts.append(text(x + 16, y + 38, title, 12.5, C_MUTED, 600))
    parts.append(text(x + 16, y + 64, value, value_size, C_TEXT, 700))
    if sub:
        parts.append(text(x + 16, y + 80, sub, 10.5, C_MUTED, 500))


def draw_table(parts, S, x, y, w):
    truth, res = S["truth"], S["res"]
    names = [os.path.splitext(f)[0] for f in truth["files"]]
    rows = [(names[i], tr) for i, tr in enumerate(res["tracks"])]
    rows.sort(key=lambda r: r[1].arrival_s if r[1].arrival_s is not None else float("inf"))
    t_first = min(tr.arrival_s for _, tr in rows if tr.arrival_s is not None)
    max_delta = max((tr.arrival_s - t_first) for _, tr in rows if tr.arrival_s is not None) or 1e-3
    max_res = max([abs(tr.residual_s or 0.0) for _, tr in rows] + [1e-5])
    parts.append(text(x, y, "Per recording", 17, C_TEXT, 700))
    parts.append(text(x, y + 18, "arrival after the first, onset over noise floor, timing residual", 12, C_MUTED, 500))
    hy = y + 40
    col_bar, bar_w = x + 62, 92
    col_snr = x + 250  # right edge of the SNR column
    res_mid = x + 290  # centre of the residual bar
    col_res = x + w    # right edge of the residual text
    parts.append(text(x, hy, "file", 10.5, C_MUTED, 700))
    parts.append(text(col_bar, hy, "arrival after first", 10.5, C_MUTED, 700))
    parts.append(text(col_snr, hy, "SNR", 10.5, C_MUTED, 700, "end"))
    parts.append(text(col_res, hy, "residual (ms)", 10.5, C_MUTED, 700, "end"))
    rh = 25
    for k, (name, tr) in enumerate(rows):
        ry = hy + 12 + k * rh
        parts.append(rect(x - 6, ry - 2, w + 12, rh - 2, 9, "#F8FBFF" if k % 2 else "#EEF4FA", None))
        parts.append(text(x, ry + 13, fit_text(name, 8), 11.5, C_TEXT, 600))
        if tr.arrival_s is None:
            parts.append(text(col_bar, ry + 13, "not used", 11, "#B91C1C", 600))
            continue
        d_ms = (tr.arrival_s - t_first) * 1000
        parts.append(rect(col_bar, ry + 5, bar_w, 9, 4.5, "#DBEAFE"))
        parts.append(rect(col_bar, ry + 5, max(4, bar_w * d_ms / (max_delta * 1000)), 9, 4.5, "#2563EB"))
        parts.append(text(col_bar + bar_w + 6, ry + 13, f"+{d_ms:.1f} ms", 10.5, "#1E293B", 600))
        parts.append(text(col_snr, ry + 13, f"{tr.snr:.0f}×", 11, "#1E293B", 600, "end"))
        r_ms = (tr.residual_s or 0.0) * 1000
        parts.append(line(res_mid, ry + 3, res_mid, ry + 16, "#CBD5E1", 1))
        span = 26 * min(1.0, abs(r_ms) / (max_res * 1000 + 1e-9))
        parts.append(rect(res_mid if r_ms >= 0 else res_mid - span, ry + 6, max(span, 1.5), 8, 4,
                          "#0F766E" if tr.weight and tr.weight >= 0.5 else "#B91C1C"))
        parts.append(text(col_res, ry + 13, f"{r_ms:+.3f}", 10.5, "#475569", 600, "end"))
    return hy + 12 + len(rows) * rh


def draw_elevation(parts, S, x, y, w, h):
    truth, XYZ, res, p = S["truth"], S["XYZ"], S["res"], S["params"]
    sol = res["solution"]
    src_x = truth["source_position_m"][0]
    src_z = truth["source_height_m"]
    parts.append(text(x, y, "Elevation (looking north)", 17, C_TEXT, 700))
    parts.append(text(x, y + 18, "camera heights, height prior, solved height vs truth", 12, C_MUTED, 500))
    bx, by, bw, bh = x, y + 30, w, h - 30
    parts.append(rect(bx, by, bw, bh, 14, "#F8FBFF", "#D5E2F0"))
    xs = np.concatenate([XYZ[:, 0], [src_x]])
    xlo, xhi = xs.min() - 6, xs.max() + 6
    ztop = max(XYZ[:, 2].max(), src_z, sol.s_xyz[2] + 2 * sol.z_std) * 1.25 + 1
    sx = lambda X: bx + 52 + (X - xlo) / (xhi - xlo) * (bw - 66)  # noqa: E731
    sz = lambda Z: by + bh - 16 - Z / ztop * (bh - 52)  # noqa: E731
    if sol.solve_z:
        z0, zs_ = p.source_z, p.source_z_sigma
        top, bot = sz(min(ztop, z0 + 2 * zs_)), sz(max(0.0, z0 - 2 * zs_))
        parts.append(rect(bx + 2, top, bw - 4, max(2, bot - top), 8, C_EST, None, 0.10))
        parts.append(text(bx + 12, by + 16, f"height prior {z0:g} ± {zs_:g} m (95% band)", 10, "#B45309", 600))
    step = 10.0 if ztop > 25 else (5.0 if ztop > 12 else 2.0)
    zt = step
    while zt < ztop * 0.92:
        parts.append(line(bx + 8, sz(zt), bx + bw - 8, sz(zt), "#CBD5E1", 1, "3 5"))
        parts.append(text(bx + 12, sz(zt) - 3, f"{zt:g} m", 9.5, C_MUTED, 500))
        zt += step
    parts.append(line(bx + 8, sz(0), bx + bw - 8, sz(0), "#64748B", 1.2))
    parts.append(text(bx + 12, sz(0) - 3, "ground", 9.5, C_MUTED, 500))
    for i, (mx, _, mz) in enumerate(XYZ):
        px, pz = sx(mx), sz(mz)
        if S["hsig"][i] > 0:
            parts.append(line(px, sz(mz - 2 * S["hsig"][i]), px, sz(mz + 2 * S["hsig"][i]), "#0369A1", 2))
        parts.append(triangle(px, pz, 6, "#0EA5E9", "#0369A1", 1.2))
    zs_est = sol.s_xyz[2]
    for alt in sol.alternatives:
        parts.append(circle(sx(alt["x"]), sz(alt["z"]), 6, "none", "#FB7185", 2))
    if sol.solve_z:
        parts.append(line(sx(src_x), sz(zs_est - 2 * sol.z_std), sx(src_x), sz(zs_est + 2 * sol.z_std), C_EST, 2.5))
    parts.append(circle(sx(src_x), sz(zs_est), 5.5, C_EST, "#FFF7ED", 1.5))
    parts.append(line(sx(src_x) - 10, sz(src_z), sx(src_x) + 10, sz(src_z), C_TRUE, 3))
    label = (f"solved {zs_est:.1f} ± {sol.z_std:.1f} m, true {src_z:g} m" if sol.solve_z
             else f"height fixed at {zs_est:g} m (true {src_z:g} m)")
    parts.append(text(bx + bw - 12, by + 16, label, 10.5, "#1E293B", 600, "end"))


# ------------------------------ page ------------------------------


def build_svg(S: dict) -> str:
    truth, res, positions = S["truth"], S["res"], S["positions"]
    sol = res["solution"]
    src = np.array(truth["source_position_m"])
    err = float(np.linalg.norm(sol.s_xy - src))
    a, b, _ = le.ellipse_from_cov2(sol.cov_xy)
    n_used, n_all = len(res["used"]), len(truth["files"])
    pairs = res["refinement"]["used_pairs"] if res["refinement"] is not None else np.array([])
    desc = positions.get("description", S["scenario"])
    ev = positions.get("event", {})
    where = (ev.get("true_location") or {}).get("description", "")

    parts = [
        f'<svg width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="t d">',
        f"<title id=\"t\">Acoustic Event Locator: {escape(S['scenario'])} result</title>",
        f"<desc id=\"d\">Estimated position {err*100:.1f} cm from the true source with a 95% ellipse of {2*a:.2f} by {2*b:.2f} m; "
        f"{n_used} of {n_all} recordings used.</desc>",
        """<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1280" y2="760" gradientUnits="userSpaceOnUse">
    <stop stop-color="#06131F"/><stop offset="1" stop-color="#132B45"/>
  </linearGradient>
  <filter id="shadow" x="-10%" y="-10%" width="120%" height="130%">
    <feDropShadow dx="0" dy="10" stdDeviation="14" flood-color="#020617" flood-opacity="0.35"/>
  </filter>
</defs>""",
        rect(0, 0, WIDTH, HEIGHT, 26, "url(#bg)"),
        circle(150, 110, 170, "#0EA5E9", None, None, 0.08),
        circle(1130, 80, 150, "#FB7185", None, None, 0.08),
        # left: dark map panel
        rect(36, 36, 760, 688, 26, "#0F172A", "#1E293B"),
        text(60, 74, "ACOUSTIC EVENT LOCATOR", 13, "#7DD3FC", 700),
        text(60, 108, fit_text(desc, 58), 26, "#F8FAFC", 700),
        text(60, 132, fit_text(f"{ev.get('type', 'event')} · {where}" if where else ev.get("type", ""), 90), 14, "#CBD5E1", 500),
        text(60, 152, "Synthetic scenario from test_data/, located by locate_event.py; everything drawn is its output", 12.5, "#93C5FD", 500),
        # right: light results panel
        rect(816, 36, 428, 688, 26, "#F3F8FD", "#D5E2F0"),
        text(840, 74, "RESULT", 13, C_MUTED, 700),
    ]
    metric_card(parts, 840, 88, 186, "Position error", f"{err*100:.1f} cm", "#2563EB", "estimate vs. true source")
    metric_card(parts, 1036, 88, 186, "95% ellipse", f"{2*a:.2f} × {2*b:.2f} m", "#0F766E", "full axes, to scale on the map", 21)
    if sol.solve_z:
        metric_card(parts, 840, 186, 186, "Height", f"{sol.s_xyz[2]:.1f} ± {sol.z_std:.1f} m", "#EA580C",
                    f"solved; true {truth['source_height_m']:g} m", 21)
    else:
        metric_card(parts, 840, 186, 186, "Height", f"{sol.s_xyz[2]:g} m", "#EA580C", f"fixed; true {truth['source_height_m']:g} m", 21)
    metric_card(parts, 1036, 186, 186, "Recordings used", f"{n_used} / {n_all}", "#7C3AED",
                f"{int(pairs.sum())}/{len(pairs)} pairs cross-correlated" if len(pairs) else "", 21)
    tbl_end = draw_table(parts, S, 840, 306, 380)
    draw_elevation(parts, S, 840, tbl_end + 26, 380, 702 - (tbl_end + 26))
    draw_map(parts, S, (96, 186, 660, 400))
    parts.append(text(60, 690, "Regenerate: python docs/generate_demo_diagram.py", 11.5, "#CBD5E1", 600, "start", MONO))
    parts.append(text(60, 707, "synthesizes the scenario with a fixed seed, runs the locator, draws what it returned", 11, "#94A3B8", 500, "start", MONO))
    parts.append("</svg>")
    return "\n".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render the README hero diagram from a locator run.")
    ap.add_argument("--scenario", default="scenario3_fireworks", choices=gen.SCENARIOS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "docs", "demo_diagram.svg"))
    args = ap.parse_args(argv)
    S = run_locator(args.scenario, args.seed)
    svg = build_svg(S)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(svg)
    sol = S["res"]["solution"]
    print(f"Wrote {args.out}: error {np.linalg.norm(sol.s_xy - np.array(S['truth']['source_position_m'])):.3f} m, "
          f"height {sol.s_xyz[2]:.2f} m, {len(S['res']['used'])}/{len(S['truth']['files'])} recordings, "
          f"alternatives {[(round(q['x'],1), round(q['y'],1), round(q['z'],1), round(q['delta_cost'],1)) for q in sol.alternatives]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
