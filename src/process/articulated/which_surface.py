#!/usr/bin/env python3
"""Ask which part's surface each triangulated anchor actually landed on.

The articulated fit's residual says how far an anchor is from where the *model*
puts it.  That conflates two very different faults: an anchor a few mm off its own
surface because the hinge angle is slightly stale, and an anchor a hundred mm away
because the flow latched onto the other part.  Only the second one explains the
108 mm residuals, and only the second one has to be fixed in the validity test
rather than in the fit.

The two independent rigid tracks give the reference: at each frame they say where
the body is and where the lid is, without using the joint at all.  So for every
moving-part anchor we can ask the question directly -- is this point nearer the lid
surface, or nearer the body surface?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def load_poses(records_path: Path) -> dict[int, np.ndarray]:
    records = json.loads(records_path.read_text(encoding="utf-8"))
    records = records["records"] if isinstance(records, dict) and "records" in records else records
    return {int(item["frame_index"]): np.asarray(item["pose_world"], dtype=np.float64)
            for item in records if item.get("pose_world") is not None}


def surface_distance(mesh: trimesh.Trimesh, pose: np.ndarray,
                     points_w: np.ndarray) -> np.ndarray:
    """Unsigned distance from world points to the mesh placed at ``pose``."""
    inverse = np.linalg.inv(pose)
    local = (inverse[:3, :3] @ points_w.T).T + inverse[:3, 3]
    return np.abs(trimesh.proximity.signed_distance(mesh, local))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--body-mesh", type=Path, required=True)
    parser.add_argument("--lid-mesh", type=Path, required=True)
    parser.add_argument("--body-records", type=Path, required=True)
    parser.add_argument("--lid-records", type=Path, required=True)
    args = parser.parse_args()

    body_mesh = trimesh.load(args.body_mesh, process=False, force="mesh")
    lid_mesh = trimesh.load(args.lid_mesh, process=False, force="mesh")
    body_poses = load_poses(args.body_records)
    lid_poses = load_poses(args.lid_records)

    print(f"{'f':>5} {'part':>6} {'n':>4} {'d_own_mm':>22} {'d_other_mm':>22} {'nearer_other':>12}")
    print(f"{'':>5} {'':>6} {'':>4} {'p50':>7}{'p90':>7}{'max':>8} {'p50':>7}{'p90':>7}{'max':>8}")
    for dump in sorted(args.dump_dir.glob("frame_*.npz")):
        frame = int(dump.stem.split("_")[1])
        if frame not in body_poses or frame not in lid_poses:
            continue
        data = np.load(dump)
        points = np.asarray(data["positions_w"], dtype=np.float64)
        parts = np.asarray(data["part_ids"], dtype=np.int64)
        to_body = surface_distance(body_mesh, body_poses[frame], points) * 1000.0
        to_lid = surface_distance(lid_mesh, lid_poses[frame], points) * 1000.0
        for part_id, name in ((0, "body"), (1, "lid")):
            mask = parts == part_id
            if not mask.any():
                continue
            own, other = (to_body, to_lid) if part_id == 0 else (to_lid, to_body)
            own, other = own[mask], other[mask]
            frac = float(np.mean(other < own)) * 100.0
            print(f"{frame:>5} {name:>6} {int(mask.sum()):>4} "
                  f"{np.percentile(own, 50):7.1f}{np.percentile(own, 90):7.1f}{own.max():8.1f} "
                  f"{np.percentile(other, 50):7.1f}{np.percentile(other, 90):7.1f}{other.max():8.1f} "
                  f"{frac:11.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
