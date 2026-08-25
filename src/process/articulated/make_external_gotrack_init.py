#!/usr/bin/env python3
"""Convert an externally supplied articulated state into GoTrack init records.

This is the initialization path for datasets that provide a calibrated 6-DoF body
pose and joint coordinate but no stereo pair or measured depth.  It deliberately
does not pretend that the pose came from FoundationPose: the source is recorded in
every per-camera JSON and the downstream run remains a tracking-only evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import load_cameras
from joint_schema import load_single_joint_spec


def load_pose(path: Path) -> np.ndarray:
    pose = np.asarray(np.load(path), dtype=np.float64)
    if pose.shape == (3, 4):
        pose = np.vstack([pose, [0.0, 0.0, 0.0, 1.0]])
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"{path}: expected a finite 3x4 or 4x4 pose, got {pose.shape}")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-7):
        raise ValueError(f"{path}: pose has an invalid homogeneous last row")
    return pose


def load_scalar(path: Path) -> float:
    value = np.asarray(np.load(path), dtype=np.float64)
    if value.size != 1 or not np.isfinite(value).all():
        raise ValueError(f"{path}: expected one finite joint coordinate, got {value.shape}")
    return float(value.reshape(-1)[0])


def make_record(
    *, frame_index: int, pose_world: np.ndarray, joint_type: str,
    joint_value: float, source: str,
) -> dict[str, object]:
    record: dict[str, object] = {
        "frame_index": int(frame_index),
        "pose_world": pose_world.tolist(),
        "joint_type": joint_type,
        "joint_value": float(joint_value),
        "source": source,
    }
    if joint_type == "revolute":
        record["theta_rad"] = float(joint_value)
        record["theta_deg"] = float(np.degrees(joint_value))
    # The same world-space state is broadcast to every view.  Positive equal scores
    # prevent the init fusion stage from discarding records that have no per-view
    # estimator confidence by construction.
    record.update({key: 1.0 for key in (
        "certainty_count_above_threshold",
        "stage3_correspondence_count",
        "inliers_ratio",
        "pose_score",
        "confidence_count_above_threshold",
    )})
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--pose-npy", type=Path, required=True)
    parser.add_argument("--joint-value-npy", type=Path, required=True)
    parser.add_argument("--joint-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", default="external articulated ground truth")
    args = parser.parse_args()

    if args.frame_index < 0:
        raise ValueError("--frame-index must be non-negative")
    pose_world = load_pose(args.pose_npy.expanduser().resolve())
    joint_value = load_scalar(args.joint_value_npy.expanduser().resolve())
    spec = load_single_joint_spec(args.joint_json.expanduser().resolve())
    lower, upper = spec.limits
    tolerance = 1.0e-6 * max(1.0, abs(lower), abs(upper))
    if not lower - tolerance <= joint_value <= upper + tolerance:
        raise ValueError(
            f"external joint value {joint_value:.8g} is outside {spec.joint_type} "
            f"range [{lower:.8g}, {upper:.8g}]")

    record = make_record(
        frame_index=args.frame_index,
        pose_world=pose_world,
        joint_type=spec.joint_type,
        joint_value=joint_value,
        source=args.source,
    )
    cameras = load_cameras(args.capture_dir.expanduser().resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    for camera_id in cameras:
        (args.out / f"{camera_id}.json").write_text(
            json.dumps([record], indent=2) + "\n", encoding="utf-8")
    manifest = {
        "source": args.source,
        "frame_index": int(args.frame_index),
        "pose_npy": str(args.pose_npy.expanduser().resolve()),
        "joint_value_npy": str(args.joint_value_npy.expanduser().resolve()),
        "joint_json": str(args.joint_json.expanduser().resolve()),
        "joint_type": spec.joint_type,
        "joint_value": joint_value,
        "camera_ids": sorted(cameras),
    }
    (args.out / "external_init_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"frame {args.frame_index}: external {spec.joint_type} value={joint_value:.8g}; "
        f"wrote {len(cameras)} camera files to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
