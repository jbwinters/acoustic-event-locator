#!/usr/bin/env python3
"""
Run locate_event.py on every generated scenario under test_data/ and score it against the
ground truth in metadata.json (same local frame). Reports position error, whether the truth
falls inside the 95% ellipse, timing residuals and, in prior mode, clock-offset errors.

Usage
  python generate_test_data.py            # once, writes the tracks and metadata
  python run_test_scenarios.py            # synchronized-clock model (default)
  python run_test_scenarios.py --clock_sigma_ms 2 --extra --min_snr 5
Exit status is 1 if any scenario exceeds --max_error_m.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIOS = ("scenario1_gunshot", "scenario2_explosion", "scenario3_fireworks", "scenario4_window_shot", "scenario5_urban_canyon")


def run_locator(scenario_dir, out_dir, extra):
    cmd = [
        sys.executable, os.path.join(HERE, "locate_event.py"),
        "--videos_dir", scenario_dir, "--positions", os.path.join(scenario_dir, "positions.json"),
        "--out", out_dir,
    ] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return r.returncode, r.stdout + r.stderr


def score(results, truth):
    est = np.array([results["event_location_local_m"]["x"], results["event_location_local_m"]["y"]])
    tru = np.array(truth["source_position_m"][:2])
    err = float(np.linalg.norm(est - tru))
    # An ambiguous geometry (e.g. cameras in a line) yields two solutions that fit equally well;
    # the locator reports both, so score the nearest reported solution as well.
    alts = [np.array([a["x"], a["y"]]) for a in results["fit"].get("alternatives", [])]
    nearest = min([err] + [float(np.linalg.norm(a - tru)) for a in alts])
    a, b, ang = (results["confidence_ellipse_95"][k] for k in ("semi_major_m", "semi_minor_m", "angle_deg"))
    d = tru - est
    ca, sa = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
    u, v = ca * d[0] + sa * d[1], -sa * d[0] + ca * d[1]
    inside = (u / max(a, 1e-9)) ** 2 + (v / max(b, 1e-9)) ** 2 <= 1.0
    offs_err = None
    if results["clock_model"]["mode"] == "prior":
        est_off = np.array([p["clock_offset_s"] or 0.0 for p in results["per_recording"] if p["used"]])
        tru_off = np.array([o for o, p in zip(truth["clock_offsets_s"], results["per_recording"]) if p["used"]])
        est_rel = est_off - est_off.mean()
        tru_rel = tru_off - tru_off.mean()
        offs_err = float(np.max(np.abs(est_rel - tru_rel)) * 1000)
    true_occ = sorted(i for i, dm in enumerate(truth.get("occlusion_detour_m", [])) if dm > 0)
    names = [p["file"] for p in results["per_recording"]]
    det_occ = sorted(names.index(f) for f in results["fit"].get("occluded", []) if f in names)
    hm = results.get("height_model", {}).get("source", {})
    z_err = float(results["event_location_local_m"]["z"] - truth.get("source_height_m", 0.0))
    z_std = float(results["position_std_m"].get("z", 0.0))
    return {
        "error_m": err, "nearest_m": nearest, "inside_95": bool(inside), "a_m": a, "b_m": b, "rmse_ms": results["fit"]["rmse_ms"],
        "z_solved": bool(hm.get("solved", False)), "z_err_m": z_err, "z_std_m": z_std,
        "true_occluded": true_occ, "detected_occluded": det_occ,
        "used": results["fit"]["recordings_used"], "total": results["fit"]["recordings_total"],
        "ambiguous": results["fit"]["ambiguous"], "offset_err_ms": offs_err, "warnings": results["warnings"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score locate_event.py on the synthetic scenarios.")
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--clock_sigma_ms", type=float, default=0.0)
    ap.add_argument("--z", choices=("truth", "prior", "free"), default="truth",
                    help="Event height: fixed at the true value, solved with the scenario's height prior, or solved freely.")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "scenarios"))
    ap.add_argument("--max_error_m", type=float, default=0.5, help="Fail if any scenario error (nearest solution when ambiguous) exceeds this.")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="Extra arguments passed to locate_event.py")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    rows = []
    for name in args.scenarios:
        sdir = os.path.join(HERE, "test_data", name)
        meta_path = os.path.join(sdir, "metadata.json")
        if not os.path.exists(meta_path):
            print(f"{name}: no metadata.json (run generate_test_data.py first)")
            continue
        truth = json.load(open(meta_path))
        missing = [f for f in truth.get("files", []) if not os.path.exists(os.path.join(sdir, f))]
        if missing:
            print(f"{name}: tracks missing {missing} (run generate_test_data.py first)")
            continue
        if args.z == "truth":
            extra = ["--source_height_m", str(truth.get("source_height_m", 0.0))]
        elif args.z == "prior":
            hp = truth.get("height_prior_m") or {"mean": 0.0, "sigma": 50.0}
            extra = ["--source_height_m", str(hp["mean"]), "--source_height_sigma_m", str(hp["sigma"])]
        else:
            extra = ["--source_height_m", "0", "--source_height_sigma_m", "1000"]
        if args.clock_sigma_ms > 0:
            extra += ["--clock_sigma_ms", str(args.clock_sigma_ms)]
        extra += args.extra
        out_dir = os.path.join(args.out, name)
        code, log = run_locator(sdir, out_dir, extra)
        if not args.quiet:
            print(f"\n=== {name} ===")
            print(log.strip())
        if code != 0:
            rows.append((name, None))
            continue
        results = json.load(open(os.path.join(out_dir, "results.json")))
        rows.append((name, score(results, truth)))

    print("\n" + "=" * 78)
    print(f"{'scenario':22s} {'error':>8s} {'nearest':>8s} {'95% ellipse':>14s} {'in':>3s} {'rmse':>8s} {'used':>5s} {'height err':>16s} {'offset err':>10s}")
    failed = False
    for name, s in rows:
        if s is None:
            print(f"{name:22s} FAILED")
            failed = True
            continue
        scored = s["nearest_m"] if s["ambiguous"] else s["error_m"]
        flag = "" if scored <= args.max_error_m else "  <-- exceeds limit"
        failed |= scored > args.max_error_m
        off = f"{s['offset_err_ms']:.2f} ms" if s["offset_err_ms"] is not None else "-"
        amb = " (ambiguous: alternative solution also reported)" if s["ambiguous"] else ""
        zcol = f"{s['z_err_m']:+6.2f} +- {s['z_std_m']:5.2f} m" if s["z_solved"] else "fixed (truth)"
        if s["true_occluded"] or s["detected_occluded"]:
            ok = s["true_occluded"] == s["detected_occluded"]
            amb += f" [occluded {len(s['detected_occluded'])}/{len(s['true_occluded'])} {'correctly identified' if ok else 'MISMATCH det ' + str(s['detected_occluded']) + ' true ' + str(s['true_occluded'])}]"
        print(f"{name:22s} {s['error_m']:6.2f} m {s['nearest_m']:6.2f} m {s['a_m']:6.2f}x{s['b_m']:5.2f} m {'y' if s['inside_95'] else 'n':>3s} {s['rmse_ms']:5.3f} ms {s['used']:>2d}/{s['total']:<2d} {zcol:>16s} {off:>10s}{amb}{flag}")
    print("error = best solution vs truth (x, y); nearest = closest of best and alternative solutions; in = truth inside 95% ellipse;"
          " height err = estimated - true event height with its 1-sigma std when solved (--z prior|free)")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
