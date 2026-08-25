#!/usr/bin/env python3
"""Compose articulated records from a rigid parent trajectory and joint anchors.

This is the recovery path for sequences where tracking the moving part corrupts or
terminates the common body pose.  The parent mesh is tracked rigidly with GoTrack,
while the one-dimensional joint coordinate is supplied by sparse stereo.  The
result has the same record layout consumed by the articulated renderer and temporal
constraint pass; no SE(3) value is estimated or modified here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from constrain_joint_trajectory import load_anchors
from common import load_articulation


def compose_records(
    body_records: list[dict],
    anchors: dict[int, float],
    *,
    body_object: str,
    articulated_object: str,
    joint_type: str = "prismatic",
    joint_records: list[dict] | None = None,
) -> list[dict]:
    usable = [record for record in body_records
              if record.get("status") == "ok" and record.get("pose_world") is not None]
    if not usable:
        raise ValueError("rigid body trajectory has no valid pose records")
    frames = np.asarray([int(record["frame_index"]) for record in usable], dtype=np.int64)
    if np.any(np.diff(frames) <= 0):
        raise ValueError("rigid body records must be strictly ordered by frame_index")

    anchor_frames = np.asarray(sorted(anchors), dtype=np.int64)
    anchor_values = np.asarray([anchors[int(frame)] for frame in anchor_frames],
                               dtype=np.float64)
    provisional = np.interp(frames, anchor_frames, anchor_values)
    joint_by_frame = None
    if joint_records is not None:
        joint_by_frame = {
            int(record["frame_index"]): float(record["joint_value"])
            for record in joint_records
            if record.get("status") == "ok" and record.get("joint_value") is not None
        }
        missing = sorted(set(map(int, frames)) - set(joint_by_frame))
        if missing:
            raise ValueError(
                f"joint source trajectory is missing {len(missing)} body frames; "
                f"first missing: {missing[:10]}")
        provisional = np.asarray([joint_by_frame[int(frame)] for frame in frames])

    output = []
    for record, joint_value in zip(usable, provisional):
        row = dict(record)
        row.update({
            "stage": "rigid_body_plus_sparse_joint",
            "object_name": articulated_object,
            "joint_type": joint_type,
            "joint_value": float(joint_value),
            "joint_value_source": ("tracked_joint_with_sparse_depth_pending"
                                   if joint_by_frame is not None
                                   else "linear_sparse_depth_initialization"),
            "body_pose_source": "rigid_parent_gotrack",
            "body_pose_source_object": body_object,
        })
        output.append(row)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-run-dir", type=Path, required=True,
                        help="GoTrack root containing <body-object>/world_pose_records.json")
    parser.add_argument("--body-object", default="body_1F")
    parser.add_argument("--object", required=True, dest="articulated_object")
    parser.add_argument("--joint-anchors", type=Path, required=True)
    parser.add_argument("--joint-run-dir", type=Path, default=None,
                        help="Optional articulated GoTrack root whose per-frame joint "
                             "values are retained outside the corrected anchor span.")
    parser.add_argument("--joint-source-object", default=None,
                        help="Object directory below --joint-run-dir. Defaults to --object.")
    parser.add_argument("--out-run-dir", type=Path, required=True,
                        help="Output root; records are written below <object>/")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_path = (args.body_run_dir / args.body_object /
                   "world_pose_records.json")
    destination_dir = args.out_run_dir / args.articulated_object
    destination_path = destination_dir / "world_pose_records.json"
    if destination_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {destination_path}; pass --overwrite")

    body_records = json.loads(source_path.read_text(encoding="utf-8"))
    anchors = load_anchors(args.joint_anchors, args.articulated_object)
    articulation = load_articulation(args.articulated_object)
    joint_records = None
    joint_record_path = None
    if args.joint_run_dir is not None:
        joint_source_object = args.joint_source_object or args.articulated_object
        joint_record_path = (args.joint_run_dir / joint_source_object /
                             "world_pose_records.json")
        joint_records = json.loads(joint_record_path.read_text(encoding="utf-8"))
    output = compose_records(
        body_records, anchors, body_object=args.body_object,
        articulated_object=args.articulated_object,
        joint_type=articulation.joint_type,
        joint_records=joint_records)

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    report = {
        "object": args.articulated_object,
        "body_object": args.body_object,
        "body_record_path": str(source_path),
        "joint_anchors": str(args.joint_anchors),
        "joint_record_path": None if joint_record_path is None else str(joint_record_path),
        "output_record_path": str(destination_path),
        "num_records": len(output),
        "first_frame": int(output[0]["frame_index"]),
        "last_frame": int(output[-1]["frame_index"]),
        "body_pose_modified": False,
        "joint_type": articulation.joint_type,
        "joint_unit": articulation.joint_unit,
    }
    report_path = args.out_run_dir / "body_joint_composition.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"composed {len(output)} records from rigid body poses and sparse joint anchors")
    print(f"wrote {destination_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
