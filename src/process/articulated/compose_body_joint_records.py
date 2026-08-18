#!/usr/bin/env python3
"""Compose articulated records from a rigid parent trajectory and joint anchors.

This is the recovery path for sequences where tracking the moving part corrupts or
terminates the common body pose.  The parent mesh is tracked rigidly with GoTrack,
while the one-dimensional prismatic coordinate is supplied by sparse stereo.  The
result has the same record layout consumed by the articulated renderer and temporal
constraint pass; no SE(3) value is estimated or modified here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from constrain_joint_trajectory import load_anchors


def compose_records(
    body_records: list[dict],
    anchors: dict[int, float],
    *,
    body_object: str,
    articulated_object: str,
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

    output = []
    for record, joint_value in zip(usable, provisional):
        row = dict(record)
        row.update({
            "stage": "rigid_body_plus_sparse_joint",
            "object_name": articulated_object,
            "joint_type": "prismatic",
            "joint_value": float(joint_value),
            "joint_value_source": "linear_sparse_stereo_initialization",
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
    output = compose_records(
        body_records, anchors, body_object=args.body_object,
        articulated_object=args.articulated_object)

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    report = {
        "object": args.articulated_object,
        "body_object": args.body_object,
        "body_record_path": str(source_path),
        "joint_anchors": str(args.joint_anchors),
        "output_record_path": str(destination_path),
        "num_records": len(output),
        "first_frame": int(output[0]["frame_index"]),
        "last_frame": int(output[-1]["frame_index"]),
        "body_pose_modified": False,
    }
    report_path = args.out_run_dir / "body_joint_composition.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"composed {len(output)} records from rigid body poses and sparse joint anchors")
    print(f"wrote {destination_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
