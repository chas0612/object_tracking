#!/usr/bin/env python3
"""Merge a forward and a reverse GoTrack pass into one trajectory.

Tracking carries state from each frame to the next one it visits, so a pass only
covers the frames on one side of its seed. Running the video backwards recovers the
other side from the same init, and the two spans meet at the seed frame -- which
both passes solve independently, from the same external pose, and which is
therefore the merge's own consistency check rather than a frame to be picked.

The output is written where a single-pass run would have written it, so the video
renderer and every reader downstream take a merged run without being told.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _records(run_dir: Path, object_name: str) -> list[dict]:
    path = run_dir / object_name / "world_pose_records.json"
    if not path.is_file():
        raise FileNotFoundError(f"No world_pose_records.json under {run_dir / object_name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _pose(record: dict) -> np.ndarray | None:
    """A record's world pose as 4x4, whichever way it stored it."""
    if record.get("pose_world") is not None:
        return np.asarray(record["pose_world"], dtype=np.float64)
    if record.get("rotation_world") is None or record.get("translation_world_m") is None:
        return None
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(record["rotation_world"], dtype=np.float64)
    pose[:3, 3] = np.asarray(record["translation_world_m"], dtype=np.float64)
    return pose


def _disagreement(a: dict, b: dict) -> dict:
    pose_a, pose_b = _pose(a), _pose(b)
    entry: dict = {"frame_index": int(a["frame_index"])}
    if pose_a is not None and pose_b is not None:
        delta = np.linalg.inv(pose_a) @ pose_b
        entry["body_rotation_deg"] = float(np.degrees(np.arccos(
            np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))))
        entry["body_translation_mm"] = float(np.linalg.norm(delta[:3, 3]) * 1000.0)
    if a.get("joint_value") is not None and b.get("joint_value") is not None:
        entry["joint_value_delta"] = float(b["joint_value"] - a["joint_value"])
    if a.get("theta_deg") is not None and b.get("theta_deg") is not None:
        entry["theta_delta_deg"] = float(b["theta_deg"] - a["theta_deg"])
    return entry


def merge(forward: list[dict], reverse: list[dict]) -> tuple[list[dict], dict]:
    """One record per frame, each tagged with the pass it came from.

    Where only one pass solved a frame the choice is forced. Where both did -- the
    seed, and any frame a pass reached from the far side -- forward is kept and the
    difference is measured rather than averaged: these are two tracks that started
    from the same pose and accumulated their own drift, so the gap between them is
    evidence about the run, and an average would destroy exactly that.
    """
    by_frame: dict[int, dict] = {}
    for source, records in (("forward", forward), ("reverse", reverse)):
        for record in records:
            index = int(record["frame_index"])
            slot = by_frame.setdefault(index, {})
            slot[source] = record

    merged: list[dict] = []
    overlap: list[dict] = []
    counts = {"forward": 0, "reverse": 0, "neither": 0, "both": 0}
    for index in sorted(by_frame):
        slot = by_frame[index]
        good = {name: rec for name, rec in slot.items() if rec.get("status") == "ok"}
        if len(good) == 2:
            counts["both"] += 1
            overlap.append(_disagreement(good["forward"], good["reverse"]))
        if good:
            source = "forward" if "forward" in good else "reverse"
            counts[source] += 1
            chosen = dict(good[source])
        else:
            counts["neither"] += 1
            chosen = dict(slot.get("forward") or slot["reverse"])
        chosen["merge_source"] = source if good else "none"
        chosen["merge_solved_by"] = sorted(good)
        merged.append(chosen)

    report = {"frames": len(merged), "counts": counts, "overlap": overlap}
    if overlap:
        for key in ("body_rotation_deg", "body_translation_mm", "theta_delta_deg"):
            values = np.array([abs(entry[key]) for entry in overlap if key in entry])
            if values.size:
                report[f"overlap_{key}"] = {
                    "median": float(np.median(values)), "max": float(values.max())}
    return merged, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-dir", type=Path, required=True)
    parser.add_argument("--reverse-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Written as <out-dir>/<object>/world_pose_records.json, the "
                             "layout a single pass produces.")
    parser.add_argument("--object", required=True)
    args = parser.parse_args()

    forward = _records(args.forward_dir, args.object)
    reverse = _records(args.reverse_dir, args.object)
    merged, report = merge(forward, reverse)

    out = args.out_dir / args.object
    out.mkdir(parents=True, exist_ok=True)
    (out / "world_pose_records.json").write_text(
        json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "bidirectional_merge.json").write_text(
        json.dumps({"forward_dir": str(args.forward_dir),
                    "reverse_dir": str(args.reverse_dir), **report}, indent=2) + "\n",
        encoding="utf-8")

    counts = report["counts"]
    print(f"merged {report['frames']} frames: {counts['forward']} forward, "
          f"{counts['reverse']} reverse, {counts['both']} solved by both, "
          f"{counts['neither']} by neither")
    for key in ("body_rotation_deg", "body_translation_mm", "theta_delta_deg"):
        stats = report.get(f"overlap_{key}")
        if stats:
            print(f"  overlap |{key}|: median {stats['median']:.4f}  max {stats['max']:.4f}")
    print(f"wrote {out / 'world_pose_records.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
