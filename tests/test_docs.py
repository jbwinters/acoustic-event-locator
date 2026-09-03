"""The README diagram generator must keep working against the current data formats."""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "docs"))

import generate_demo_diagram as demo  # noqa: E402


def test_demo_diagram_builds_from_a_real_run(tmp_path):
    S = demo.run_locator("scenario3_fireworks", seed=0)
    sol = S["res"]["solution"]
    assert np.linalg.norm(sol.s_xy - np.array(S["truth"]["source_position_m"])) < 0.1
    svg = demo.build_svg(S)
    root = ET.fromstring(svg)  # well-formed XML
    assert root.tag.endswith("svg")
    for needle in ("Zoom on the estimate", "Per recording", "Elevation (looking north)", "cam1", "Position error"):
        assert needle in svg
    assert "nan" not in svg.lower().replace("nanosecond", "")
    out = tmp_path / "demo.svg"
    assert demo.main(["--scenario", "scenario4_window_shot", "--out", str(out)]) == 0
    assert out.stat().st_size > 10000 and "solved" in out.read_text()
