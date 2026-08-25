#!/usr/bin/env python3
"""Small dependency-free checks for external articulated initialization."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/process/articulated"))

from make_external_gotrack_init import load_pose, load_scalar, make_record  # noqa: E402


failures = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global failures
    if condition:
        print(f"  PASS  {label}")
    else:
        failures += 1
        print(f"  FAIL  {label}  {detail}")


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    pose_path = root / "pose.npy"
    joint_path = root / "joint.npy"
    pose = np.eye(4)
    pose[:3, 3] = [0.1, -0.2, 0.8]
    np.save(pose_path, pose)
    np.save(joint_path, np.asarray(0.75))

    loaded_pose = load_pose(pose_path)
    loaded_joint = load_scalar(joint_path)
    check("4x4 world pose round-trips", np.array_equal(loaded_pose, pose))
    check("scalar joint coordinate round-trips", loaded_joint == 0.75)

    record = make_record(
        frame_index=10,
        pose_world=loaded_pose,
        joint_type="revolute",
        joint_value=loaded_joint,
        source="synthetic test",
    )
    check("record keeps the seed frame", record["frame_index"] == 10)
    check("revolute seed carries radians", record["theta_rad"] == 0.75)
    check(
        "revolute seed carries degrees",
        np.isclose(record["theta_deg"], np.degrees(0.75)),
    )
    check(
        "broadcast init has a positive fusion score",
        record["certainty_count_above_threshold"] > 0,
    )

    prismatic = make_record(
        frame_index=4,
        pose_world=loaded_pose,
        joint_type="prismatic",
        joint_value=0.12,
        source="synthetic test",
    )
    check("prismatic seed keeps neutral joint_value", prismatic["joint_value"] == 0.12)
    check("prismatic seed is never labelled as degrees", "theta_deg" not in prismatic)

print(f"\n{failures} failures")
raise SystemExit(1 if failures else 0)
