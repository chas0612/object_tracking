#!/usr/bin/env python3
"""Small executable checks for sparse joint trajectory constraints."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/process/articulated"))

from constrain_joint_trajectory import (  # noqa: E402
    constrain_records,
    solve_soft_trajectory,
    solve_trajectory,
)


failures = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global failures
    if condition:
        print(f"  PASS  {label}" + (f"  {detail}" if detail else ""))
    else:
        failures += 1
        print(f"  FAIL  {label}" + (f"  {detail}" if detail else ""))


print("hard stereo constraints propagate through the gap")
frames = np.arange(0, 8)
raw = np.array([0.0, 0.0, 0.25, 0.28, 0.29, 0.21, 0.0, 0.0])
answer = solve_trajectory(frames, raw, {1: 0.0, 5: 0.20})
expected = np.array([0.0, 0.0, 0.05, 0.10, 0.15, 0.20, 0.20, 0.20])
check("minimum-velocity solution is linear between anchors",
      np.allclose(answer, expected, atol=1e-10), str(answer))
check("frames outside the measured span hold the nearest anchor",
      answer[0] == 0.0 and answer[-1] == 0.20)

print("record rewrite changes only the joint coordinate")
pose = np.eye(4).tolist()
records = [
    {"frame_index": i, "status": "ok", "joint_type": "prismatic",
     "joint_value": float(raw[i]), "pose_world": pose}
    for i in range(len(raw))
]
updated, report = constrain_records(records, {1: 0.0, 5: 0.20})
check("body poses are byte-for-byte unchanged",
      all(a["pose_world"] == b["pose_world"] for a, b in zip(records, updated)))
check("raw GoTrack value is retained",
      updated[3]["joint_value_before_temporal_constraint"] == raw[3])
check("stereo anchors are exact", report["max_anchor_error"] == 0.0)
check("every solved frame is marked", all(
    record.get("joint_temporal_constraint_applied") for record in updated))

print("the same scalar constraint supports a revolute trajectory")
angular_records = [
    {"frame_index": i, "status": "ok", "joint_type": "revolute",
     "joint_value": float(np.radians([0, 0, 80, 100, -60, 20, 20, 20][i])),
     "pose_world": pose}
    for i in range(len(raw))
]
angular, angular_report = constrain_records(
    angular_records, {1: np.radians(10.0), 5: np.radians(30.0)})
check("angular interpolation stays in radians",
      np.isclose(np.degrees(angular[3]["joint_value"]), 20.0),
      f'{np.degrees(angular[3]["joint_value"]):.3f} deg')
check("revolute body poses are unchanged",
      all(a["pose_world"] == b["pose_world"]
          for a, b in zip(angular_records, angular)))
check("report records the revolute type", angular_report["joint_type"] == "revolute")

targeted, targeted_report = constrain_records(
    angular_records, {2: np.radians(10.0), 5: np.radians(30.0)},
    outside_mode="raw")
check("targeted correction preserves raw values before its first anchor",
      targeted[0]["joint_value"] == angular_records[0]["joint_value"])
check("targeted correction preserves raw values after its last anchor",
      targeted[7]["joint_value"] == angular_records[7]["joint_value"])
check("targeted correction changes the interior",
      targeted[3]["joint_value"] != angular_records[3]["joint_value"])
check("targeted report records raw outside mode",
      targeted_report["outside_mode"] == "raw")

print("idempotent input provenance")
twice, _ = constrain_records(updated, {1: 0.0, 5: 0.20})
check("a second pass preserves the original raw value",
      twice[3]["joint_value_before_temporal_constraint"] == raw[3])
check("a second pass produces the same constrained value",
      twice[3]["joint_value"] == updated[3]["joint_value"])

print("soft stereo factors and an observed upper stop")
frames = np.arange(0, 31)
anchors = {0: 0.00, 5: 0.08, 10: 0.16, 15: 0.19,
           20: 0.21, 25: 0.23, 30: 0.21}
soft, soft_report = solve_soft_trajectory(
    frames, anchors, acceleration_weight=30.0, huber_delta=0.01,
    upper_plateau_tolerance=0.025, plateau_weight=30.0,
    joint_min=0.0, joint_max=0.30)
check("near-maximum measurements form a contiguous stop plateau",
      soft_report["plateau_frames"] == [20, 25, 30],
      str(soft_report["plateau_frames"]))
check("a biased-low stop measurement is lifted",
      soft[20] > anchors[20] + 0.01, f"{soft[20]:.4f} m")
check("the plateau remains close to the observed maximum",
      min(soft[20], soft[25], soft[30]) > 0.225,
      str([round(soft[i], 4) for i in (20, 25, 30)]))
check("joint bounds are respected", soft.min() >= 0.0 and soft.max() <= 0.30)

print(f"\n{failures} failures")
raise SystemExit(1 if failures else 0)
