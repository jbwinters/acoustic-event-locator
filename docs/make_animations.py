#!/usr/bin/env python3
"""
Render the README animations from real pipeline runs.

  python docs/make_animations.py                       # everything into docs/anim/
  python docs/make_animations.py --only how_it_works_2d scenario4_window_shot

Every frame is computed from the same synthetic generator and locator code the tests use:
the wavefront is drawn from the true source, the waveform strips are the band-passed audio the
locator analyzed, the picks, estimate, ellipse and alternative solutions come out of
locate_event.locate_from_signals / solve_tdoa.
"""

import argparse
import os
import shutil
import sys
import tempfile
import textwrap

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle, Ellipse  # noqa: E402
import soundfile as sf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import locate_event as le  # noqa: E402
import generate_test_data as gen  # noqa: E402

FS = 48000
C_MIC, C_HIT, C_EST, C_TRUE, C_ALT = "#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#d62728"


# ------------------------------ shared drawing helpers ------------------------------


def ellipse_patch(sol, **kw):
    a, b, ang = le.ellipse_from_cov2(sol.cov_xy)
    return Ellipse((sol.s_xy[0], sol.s_xy[1]), 2 * a, 2 * b, angle=ang, fill=False, **kw)


def set_map_limits(ax, XY, extra_pts=(), pad=8.0):
    pts = np.vstack([XY[:, :2]] + [np.asarray(p, float)[None, :2] for p in extra_pts])
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    span = max(hi - lo)
    mid = 0.5 * (lo + hi)
    ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
    ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
    ax.set_aspect("equal", adjustable="box")


def draw_strips(ax, xf_list, t_lo, t_hi, labels, picks=None):
    """Normalized, vertically stacked band-passed waveforms in [t_lo, t_hi]. Returns
    (cursor_line, pick_lines) where pick_lines start invisible."""
    n = len(xf_list)
    a, b = int(t_lo * FS), int(t_hi * FS)
    tt = np.arange(a, b) / FS
    pick_lines = []
    for i, xf in enumerate(xf_list):
        seg = xf[a:b]
        seg = seg / (np.max(np.abs(seg)) + 1e-12) * 0.42
        y0 = n - 1 - i
        ax.plot(tt, y0 + seg, lw=0.6, color="0.25")
        ax.text(t_lo, y0 + 0.45, labels[i], fontsize=7, va="bottom", ha="left", color="0.3")
        if picks is not None and picks[i] is not None:
            ln = ax.axvline(picks[i], ymin=(y0) / n, ymax=(y0 + 1) / n, color=C_HIT, lw=1.2, visible=False)
            pick_lines.append(ln)
        else:
            pick_lines.append(None)
    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.set_xlabel("time in recording (s)")
    ax.grid(True, axis="x", alpha=0.25)
    cursor = ax.axvline(t_lo, color=C_HIT, lw=1.0, alpha=0.8)
    return cursor, pick_lines


def prepare_tracks(tracks, band=(200.0, 4000.0)):
    return [le.apply_bandpass(x, FS, band[0], band[1]) for x in tracks]


# ------------------------------ how it works (2D explainer) ------------------------------


def render_explainer(out_path, fps=15, dpi=80):
    XYZ = np.array([[0, 0, 1.5], [20, 0, 2.0], [20, 20, 1.0], [0, 20, 3.0]], float)
    src = np.array([4.0, 7.0, 0.0])
    c = le.speed_of_sound_mps(20.0)
    tracks, truth = gen.synthesize_scenario(XYZ, src, c, "gunshot", noise_rms=0.003, rng=np.random.default_rng(0))
    xf = prepare_tracks(tracks)
    p = le.PipelineParams(source_z=0.0)
    res = le.locate_from_signals(tracks, FS, XYZ, c, p)
    sol = res["solution"]
    arr = np.array([tr.arrival_s for tr in res["tracks"]])
    trace = []
    sol_tr = le.solve_tdoa(arr, XYZ, c, sigma_t=sol.sigma_t, source_z=0.0, trace=trace)
    path = np.array([th[:2] for th in trace])
    t_emit = truth["emission_time_s"]
    t_true = np.array(truth["arrival_times_s"])
    tau_max = (t_true.max() - t_emit) * 1.15

    # profiled cost surface for the third act
    xs = np.linspace(-12, 32, 111)
    ys = np.linspace(-12, 32, 111)
    GX, GY = np.meshgrid(xs, ys, indexing="ij")
    D = np.sqrt((GX[..., None] - XYZ[:, 0]) ** 2 + (GY[..., None] - XYZ[:, 1]) ** 2 + (0 - XYZ[:, 2]) ** 2) / c
    yv = arr - D
    cost = np.sum((yv - yv.mean(-1, keepdims=True)) ** 2, axis=-1)
    logcost = np.log10(cost + 1e-12)

    fig = plt.figure(figsize=(11, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1])
    ax = fig.add_subplot(gs[0, 0])
    axw = fig.add_subplot(gs[0, 1])
    set_map_limits(ax, XYZ, [src], pad=12)
    ax.set_xlabel("x east (m)")
    ax.set_ylabel("y north (m)")
    ax.grid(True, alpha=0.25)
    mics = ax.scatter(XYZ[:, 0], XYZ[:, 1], marker="^", s=90, c=[C_MIC] * 4, zorder=5)
    for i, (x, y, _) in enumerate(XYZ):
        ax.annotate(f"mic {i+1}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.scatter([src[0]], [src[1]], marker="*", s=160, c=C_TRUE, zorder=6, label="true source")
    wave = Circle((src[0], src[1]), 0.0, fill=False, color=C_HIT, lw=1.5)
    ax.add_patch(wave)
    title = ax.set_title("1. A wavefront spreads at the speed of sound; each microphone hears it at a different time", fontsize=9)
    t_lo, t_hi = t_true.min() - 0.008, t_true.max() + 0.025
    cursor, pick_lines = draw_strips(axw, xf, t_lo, t_hi, [f"mic {i+1}" for i in range(4)], picks=list(arr))
    axw.set_title("band-passed audio of each recording", fontsize=9)

    fig.tight_layout()
    hyper_artists, cost_im, path_line, path_pts, ell, est_pt, starts_pts = [], [None], [None], [None], [None], [None], [None]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    n1, n2, n3, hold = 45, 36, 40, 18
    total = n1 + n2 + n3 + hold

    def frame(k):
        if k < n1:
            tau = tau_max * (k + 1) / n1
            wave.set_radius(c * tau)
            cursor.set_xdata([t_emit + tau])
            hit = (t_true - t_emit) <= tau
            mics.set_color([C_HIT if h else C_MIC for h in hit])
            for i, ln in enumerate(pick_lines):
                if ln is not None:
                    ln.set_visible(hit[i])
            return
        if k < n1 + n2:
            j = (k - n1) // 6
            if j < len(pairs) and len(hyper_artists) <= j:
                i, m = pairs[j]
                F = (D[..., i] - D[..., m]) - (arr[i] - arr[m])
                cs = ax.contour(GX, GY, F, levels=[0.0], colors=["#9467bd"], linewidths=1.2)
                hyper_artists.append(cs)
                title.set_text(f"2. Each pair's arrival-time difference confines the source to a hyperbola (pair {i+1}-{m+1})")
            return
        if k < n1 + n2 + n3:
            j = k - n1 - n2
            if cost_im[0] is None:
                for cs in hyper_artists:
                    cs.remove()
                cost_im[0] = ax.pcolormesh(GX, GY, logcost, shading="auto", cmap="viridis", alpha=0.75, zorder=0)
                wave.set_visible(False)
                mics.set_zorder(6)
                title.set_text("3. Grid search over the misfit surface, then Levenberg-Marquardt refines to the minimum")
                starts_pts[0] = ax.scatter([path[0, 0]], [path[0, 1]], marker="s", s=50, c="white", edgecolors="k", zorder=7, label="best grid start")
                path_line[0], = ax.plot([], [], "-", color="white", lw=1.5, zorder=7)
                path_pts[0] = ax.scatter([], [], s=18, c="white", edgecolors="k", zorder=8)
            npts = min(len(path), 1 + int(j * len(path) / max(1, n3 - 12)))
            path_line[0].set_data(path[:npts, 0], path[:npts, 1])
            path_pts[0].set_offsets(path[:npts])
            if j >= n3 - 12 and ell[0] is None:
                ell[0] = ellipse_patch(sol, color=C_EST, lw=2.0, zorder=9)
                ax.add_patch(ell[0])
                est_pt[0] = ax.scatter([sol.s_xy[0]], [sol.s_xy[1]], marker="*", s=200, c=C_EST, edgecolors="k", zorder=10, label="estimate")
                err = np.linalg.norm(sol.s_xy - src[:2])
                a, b, _ = le.ellipse_from_cov2(sol.cov_xy)
                title.set_text(f"4. Estimate with its 95% ellipse ({2*a:.2f} x {2*b:.2f} m)\nerror to the true source: {err*100:.1f} cm")
                ax.legend(loc="upper right", fontsize=8)
            return

    anim = FuncAnimation(fig, frame, frames=total, interval=1000 / fps)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out_path


# ------------------------------ per-scenario demonstrations ------------------------------


def run_scenario(name, seed=0):
    sdir = os.path.join(ROOT, "test_data", name)
    tmp = tempfile.mkdtemp()
    try:
        d = os.path.join(tmp, name)
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
    if hp:
        p = le.PipelineParams(source_z=hp["mean"], source_z_sigma=hp["sigma"])
    else:
        p = le.PipelineParams(source_z=truth["source_height_m"])
    res = le.locate_from_signals(tracks, FS, XYZ, c, p, height_sigma=hsig)
    return dict(name=name, truth=truth, XYZ=XYZ, hsig=hsig, c=c, tracks=tracks, res=res, params=p,
                labels=[os.path.splitext(f)[0] for f in truth["files"]], description=J.get("description", name))


def render_scenario(name, out_path, fps=15, dpi=80, seed=0):
    S = run_scenario(name, seed)
    truth, XYZ, res, c = S["truth"], S["XYZ"], S["res"], S["c"]
    sol = res["solution"]
    src = np.array(truth["source_position_m"] + [truth["source_height_m"]])
    t_emit = truth["emission_time_s"]
    t_true = np.array(truth["arrival_times_s"])
    used = res["used"]
    arr = [tr.arrival_s for tr in res["tracks"]]
    xf = prepare_tracks(S["tracks"], S["params"].band)
    three_d = sol.solve_z or bool((S["hsig"] > 0).any())
    XYZ_true = np.array(truth["microphone_positions_m"])

    fig = plt.figure(figsize=(12, 6.4 if three_d else 5.6))
    if three_d:
        gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], height_ratios=[1.15, 1], hspace=0.5, wspace=0.22)
        ax = fig.add_subplot(gs[:, 0])
        axw = fig.add_subplot(gs[0, 1])
        axz = fig.add_subplot(gs[1, 1])
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.22)
        ax = fig.add_subplot(gs[0, 0])
        axw = fig.add_subplot(gs[0, 1])
        axz = None
    fig.suptitle(textwrap.fill(f"{name}: {S['description']}", 120), fontsize=9.5)
    fig.subplots_adjust(top=0.86, bottom=0.1, left=0.07, right=0.98)

    alts = [a for a in sol.alternatives if a["delta_cost"] <= 3.0]  # only near-equivalent explanations
    extra = [src] + [np.array([a["x"], a["y"]]) for a in alts]
    set_map_limits(ax, XYZ, extra, pad=14)
    ax.set_xlabel("x east (m)")
    ax.set_ylabel("y north (m)")
    ax.grid(True, alpha=0.25)
    occluded = [used[k] for k in sol.occluded]
    colors = [("#9467bd" if i in occluded else C_MIC) if i in used else "0.6" for i in range(len(XYZ))]
    mics = ax.scatter(XYZ[:, 0], XYZ[:, 1], marker="^", s=80, c=colors, zorder=5)
    if occluded:
        ax.scatter(XYZ[occluded, 0], XYZ[occluded, 1], marker="^", s=200, facecolors="none", edgecolors="#9467bd", lw=1.5, zorder=4,
                   label="occluded: late arrival explained as a detour")
    for i, (x, y, _) in enumerate(XYZ):
        ax.annotate(S["labels"][i], (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.scatter([src[0]], [src[1]], marker="*", s=160, c=C_TRUE, zorder=6, label="true source")
    wave = Circle((src[0], src[1]), 0.0, fill=False, color=C_HIT, lw=1.5)
    ax.add_patch(wave)
    title = ax.set_title("sound spreads from the event", fontsize=9)

    t_lo = min(t_true) - 0.01
    t_hi = max(t_true) + 0.03
    cursor, pick_lines = draw_strips(axw, xf, t_lo, t_hi, S["labels"], picks=arr)
    axw.set_title("band-passed recordings; red ticks = refined arrival picks", fontsize=9)

    if axz is not None:
        axz.axhline(0, color="k", lw=0.8, alpha=0.5)
        axz.errorbar(XYZ[:, 0], sol.mic_heights, yerr=2 * sol.mic_height_std, fmt="^", ms=7, capsize=3,
                     color=C_MIC, label="cameras (height, 95%)")
        if (S["hsig"] > 0).any():
            axz.scatter(XYZ_true[:, 0], XYZ_true[:, 2], marker="_", s=120, c=C_TRUE, label="true camera heights")
        if sol.solve_z:
            zp = (S["params"].source_z, S["params"].source_z_sigma)
            axz.axhspan(max(0, zp[0] - 2 * zp[1]), zp[0] + 2 * zp[1], color=C_EST, alpha=0.10, label="height prior (95%)")
        axz.scatter([src[0]], [src[2]], marker="*", s=160, c=C_TRUE, zorder=5)
        axz.set_xlabel("x east (m)")
        axz.set_ylabel("height (m)")
        axz.set_title("elevation view", fontsize=9)
        axz.grid(True, alpha=0.25)
        axz.set_xlim(ax.get_xlim())
        zmax = max(XYZ[:, 2].max(), src[2], sol.s_xyz[2] + 2 * sol.z_std) * 1.25 + 2
        axz.set_ylim(-1, zmax)
        axz.legend(fontsize=7, loc="upper right")
    tau_max = (t_true.max() - t_emit) * 1.15
    n1, n2, hold = 50, 20, 24
    total = n1 + n2 + hold
    shown = [False]

    def frame(k):
        if k < n1:
            tau = tau_max * (k + 1) / n1
            wave.set_radius(c * tau)
            cursor.set_xdata([t_emit + tau])
            hit = (t_true - t_emit) <= tau
            mics.set_color([C_HIT if (h and i in used) else colors[i] for i, h in enumerate(hit)])
            for i, ln in enumerate(pick_lines):
                if ln is not None:
                    ln.set_visible(bool(hit[i]))
            return
        if not shown[0]:
            shown[0] = True
            wave.set_visible(False)
            ax.add_patch(ellipse_patch(sol, color=C_EST, lw=2.0, zorder=9))
            ax.scatter([sol.s_xy[0]], [sol.s_xy[1]], marker="*", s=200, c=C_EST, edgecolors="k", zorder=10, label="estimate + 95% ellipse")
            if alts:
                ax.scatter([a["x"] for a in alts], [a["y"] for a in alts], marker="o", s=90,
                           facecolors="none", edgecolors=C_ALT, lw=1.5, zorder=9, label="alternative solution")
            err = np.linalg.norm(sol.s_xy - src[:2])
            near = min([err] + [np.linalg.norm(np.array([a["x"], a["y"]]) - src[:2]) for a in alts])
            a_, b_, _ = le.ellipse_from_cov2(sol.cov_xy)
            msg = f"estimate: {err:.2f} m from the true source, 95% ellipse {2*a_:.2f} x {2*b_:.2f} m"
            if sol.ambiguous:
                msg += f"\nambiguous: alternative solution also reported, nearest one {near:.2f} m from truth"
            if sol.solve_z:
                msg += f"\nheight solved: {sol.s_xyz[2]:.1f} m (true {src[2]:.1f} m), std {sol.z_std:.2f} m"
            if sol.occluded:
                det = ", ".join(f"{S['labels'][used[k]]} +{sol.detour_m[k]:.0f} m" for k in sol.occluded)
                msg += f"\noccluded (late, down-weighted): {det}"
            title.set_text(msg)
            title.set_fontsize(8.5)
            fig.subplots_adjust(top=0.86 - 0.025 * max(0, msg.count("\n") - 1), bottom=0.17)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=8, frameon=False)
            if axz is not None:
                axz.errorbar([sol.s_xy[0]], [sol.s_xyz[2]], yerr=[[2 * sol.z_std], [2 * sol.z_std]], fmt="*", ms=14,
                             capsize=4, color=C_EST, zorder=6, label="estimated height (95%)")
                axz.legend(fontsize=7, loc="upper right")

    anim = FuncAnimation(fig, frame, frames=total, interval=1000 / fps)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render README animations from real pipeline runs.")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "anim"))
    ap.add_argument("--only", nargs="+", default=None, help="Subset of: how_it_works_2d " + " ".join(gen.SCENARIOS))
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--dpi", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    wanted = args.only or (["how_it_works_2d"] + list(gen.SCENARIOS))
    for w in wanted:
        path = os.path.join(args.out, f"{w}.gif")
        if w == "how_it_works_2d":
            render_explainer(path, fps=args.fps, dpi=args.dpi)
        else:
            render_scenario(w, path, fps=args.fps, dpi=args.dpi, seed=args.seed)
        print(f"wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
