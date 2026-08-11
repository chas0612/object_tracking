#!/usr/bin/env python3
"""Regression checks for legacy and promoted single-joint mesh metadata."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/process/articulated"))

from common import theta_grid  # noqa: E402
from joint_schema import load_single_joint_spec  # noqa: E402


OBJ = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"


def _mesh(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OBJ, encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _mesh(root / "parts/base/base.obj")
        _mesh(root / "parts/handle/handle.obj")
        generic = {
            "parts": {"3": "base", "8": "handle"},
            "joints": [{
                "type": "revolute", "parent": 3, "child": 8,
                "axis": [0, 0, 1], "origin": [0, 0, 0],
                "range_deg": [-80, 90],
            }],
        }
        path = root / "joint.json"
        path.write_text(json.dumps(generic), encoding="utf-8")
        spec = load_single_joint_spec(path)
        assert spec.part_names == ("base", "handle")
        assert spec.parent_id == 3 and spec.child_id == 8
        assert np.allclose(spec.limits, np.radians([-80, 90]))

        generic["joints"].append(dict(generic["joints"][0]))
        path.write_text(json.dumps(generic), encoding="utf-8")
        try:
            load_single_joint_spec(path)
        except ValueError as error:
            assert "exactly one joint" in str(error)
        else:
            raise AssertionError("multiple joints must be rejected")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _mesh(root / "parts/body.obj")
        _mesh(root / "parts/lid.obj")
        path = root / "joint.json"
        path.write_text(json.dumps({"measured": {
            "axis": [0, 1, 0], "origin": [0, 0, 0], "range_rad": [0, 3.6],
        }}), encoding="utf-8")
        spec = load_single_joint_spec(path)
        assert spec.part_names == ("body", "lid")
        assert spec.limits == (0.0, 3.6)

    signed = np.degrees(theta_grid(np.radians(-79.8), np.radians(88.7), 15))
    assert np.count_nonzero(np.isclose(signed, 0.0)) == 1
    assert signed.min() < 0 < signed.max()
    legacy = np.degrees(theta_grid(0.0, np.radians(205.807916), 15))
    assert np.allclose(legacy, np.arange(0.0, 196.0, 15.0))
    print("articulation schema: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
