#!/usr/bin/env python3
"""Split one articulated seed into independent parent and moving-part rigid seeds.

Diagnostic only.  The articulated tracker fits both parts in one Kabsch solve, so a
failure cannot be attributed to a part from its output alone.  Tracking each part as
an ordinary rigid object from the same frame, the same cameras and the same answer
isolates that: the parent seed is the fitted pose as-is, and the moving-part seed is
that pose composed with the joint at the fitted angle, exactly as in the articulated
renderer.

The two runs share no state, so whatever they disagree about is the coupling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import load_cameras
from joint_schema import load_single_joint_spec

SCORE_KEYS = (
    "certainty_count_above_threshold",
    "stage3_correspondence_count",
    "inliers_ratio",
    "pose_score",
    "confidence_count_above_threshold",
)


def joint_transform(
    axis: np.ndarray,
    origin: np.ndarray,
    value: float,
    joint_type: str = "revolute",
) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    if joint_type == "prismatic":
        transform = np.eye(4)
        transform[:3, 3] = axis * float(value)
        return transform
    if joint_type != "revolute":
        raise ValueError(f"unsupported joint type {joint_type!r}")
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    rotation = (np.eye(3) + np.sin(value) * cross
                + (1.0 - np.cos(value)) * (cross @ cross))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(origin, dtype=np.float64) - rotation @ origin
    return transform


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--joint-json", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=40)
    parser.add_argument("--hybrid-result", type=Path, default=None,
                        help="Explicit hybrid_result.json. Defaults to the legacy "
                             "<capture>/articulated_probe layout.")
    parser.add_argument("--repeat-frames", type=int, default=0)
    parser.add_argument("--out-root", type=Path, default=None,
                        help="Parent of gotrack_init_<part>. Defaults to capture-dir.")
    args = parser.parse_args()

    hybrid_result = args.hybrid_result
    if hybrid_result is None:
        probe = args.capture_dir / "articulated_probe" / f"frame_{args.frame_index:06d}"
        hybrid_result = probe / "hybrid/hybrid_result.json"
    record = json.loads(hybrid_result.read_text(encoding="utf-8"))
    answer = record.get("answer") or max(
        (record.get("starts") or record["results"]).values(),
        key=lambda r: r["silhouette_iou"])
    pose_body = np.asarray(answer["pose_body"], dtype=np.float64)

    joint = load_single_joint_spec(args.joint_json)
    if joint.joint_type == "revolute":
        # Current hybrid results expose the joint-type-neutral top-level value in
        # native units (radians here).  Keep the legacy fields as fallbacks so old
        # case-study seeds remain readable.
        if record.get("joint_value") is not None:
            value = float(record["joint_value"])
        elif answer.get("theta_deg") is not None:
            value = float(np.radians(answer["theta_deg"]))
        else:
            value = float(np.radians(answer["joint_disp"]))
        display = f"theta {np.degrees(value):.2f} deg"
    elif joint.joint_type == "prismatic":
        # The top-level neutral field is in the mesh's native metres; joint_disp in
        # the diagnostic answer is display millimetres and must not be composed.
        value = float(record["joint_value"])
        display = f"displacement {value * 1000.0:.1f} mm"
    else:
        raise ValueError(f"unsupported joint type {joint.joint_type!r}")
    pose_lid = pose_body @ joint_transform(
        joint.axis, joint.origin, value, joint.joint_type)

    cameras = load_cameras(args.capture_dir)
    frames = sorted({args.frame_index, *range(max(0, int(args.repeat_frames)))})
    out_root = args.capture_dir if args.out_root is None else args.out_root
    for name, pose in zip(joint.part_names, (pose_body, pose_lid)):
        out_dir = out_root / f"gotrack_init_{name}" / f"frame_{args.frame_index:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # No ``theta_deg`` here on purpose: these are rigid seeds, and a leftover angle
        # in a rigid record would be a claim the run cannot honour.
        records = [{
            "frame_index": int(frame),
            "pose_world": pose.tolist(),
            "source": f"articulated_probe frame {args.frame_index} ({name} only)",
            "silhouette_iou": float(answer["silhouette_iou"]),
            **{key: 1.0 for key in SCORE_KEYS},
        } for frame in frames]
        for camera_id in cameras:
            (out_dir / f"{camera_id}.json").write_text(
                json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {len(cameras)} cameras -> {out_dir}", flush=True)

    print(f"seed frame {args.frame_index}: {display}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
