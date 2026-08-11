#!/usr/bin/env python3
"""Promote one reviewed GoTrack pose at each CORL grasp snapshot frame.

The scheduler trajectory remains the source of truth.  Each episode's
``grasp_snapshot/selection.json`` supplies the exact encoded-video frame.  No
nearest-frame or boundary fill is allowed: a missing pose is an audit failure.
The promoted NPZ follows the static-snapshot convention and stores the selected
pose as ``frame_0`` while the manifest preserves its original frame index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, pose: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, frame_0=pose.astype(np.float32))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _latest_completed(schedules: list[Path]) -> dict[str, tuple[str, dict[str, Any]]]:
    latest: dict[str, tuple[str, dict[str, Any]]] = {}
    for schedule in schedules:
        for path in sorted((schedule / "tasks").glob("*.json")):
            task = _read_json(path)
            if (
                task.get("status") == "completed"
                and task.get("episode_rel")
                and task.get("attempt_dir")
            ):
                latest[str(task["episode_rel"])] = (schedule.name, task)
    return latest


def _pose_at(records_path: Path, selected_frame: int) -> np.ndarray:
    records = _read_json(records_path)
    if not isinstance(records, list):
        raise ValueError(f"Expected record list: {records_path}")
    matches = [
        row for row in records
        if isinstance(row, dict) and row.get("frame_index") == selected_frame
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one record at frame {selected_frame}, got {len(matches)}: {records_path}"
        )
    if matches[0].get("pose_world") is None:
        raise ValueError(f"Pose is missing at selected frame {selected_frame}: {records_path}")
    pose = np.asarray(matches[0]["pose_world"], dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"Invalid pose at selected frame {selected_frame}: {pose.shape}")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError(f"Invalid homogeneous row at frame {selected_frame}: {records_path}")
    return pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-id", nargs="+", required=True)
    parser.add_argument(
        "--runs-root-rel",
        default="object_tracking/campaigns/corl_rebuttal/allegro_v5_grasp_gotrack_runs",
    )
    parser.add_argument("--shared-root-rel", default="shared_data")
    parser.add_argument("--target-root-rel", default="capture/corl_rebuttal")
    parser.add_argument(
        "--objects", nargs="*", default=["knife_sharpener", "mug_holder"],
        help="Canonical object names to promote.",
    )
    parser.add_argument("--output-name", default="object_6d_pose_v2.npz")
    parser.add_argument(
        "--manifest-rel",
        default="capture/corl_rebuttal/object_6d_pose_dynamic_manifest.json",
    )
    parser.add_argument("--expected-tasks", type=int, default=None)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shared = Path.home() / args.shared_root_rel
    target_root = (shared / args.target_root_rel).resolve()
    runs_root = shared / args.runs_root_rel
    schedules = [runs_root / schedule_id for schedule_id in args.schedule_id]
    missing = [str(path) for path in schedules if not (path / "manifest.json").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing schedules: {missing}")
    selected_objects = set(args.objects)
    latest = {
        episode_rel: entry
        for episode_rel, entry in _latest_completed(schedules).items()
        if str(entry[1].get("object_name")) in selected_objects
    }
    if args.expected_tasks is not None and len(latest) != args.expected_tasks:
        raise RuntimeError(f"Expected {args.expected_tasks} tasks, found {len(latest)}")

    records: list[dict[str, Any]] = []
    existing: list[Path] = []
    poses: dict[str, np.ndarray] = {}
    for episode_rel, (schedule_id, task) in sorted(latest.items()):
        episode = (shared / episode_rel).resolve()
        if episode != target_root and target_root not in episode.parents:
            raise ValueError(f"Episode outside target root: {episode}")
        selection_path = episode / "grasp_snapshot/selection.json"
        selection = _read_json(selection_path)
        selected_frame = selection.get("selected_frame")
        if not isinstance(selected_frame, int) or selected_frame < 0:
            raise ValueError(f"Invalid selected_frame in {selection_path}: {selected_frame!r}")
        records_path = (
            shared / str(task["attempt_dir"]) / "gotrack_tracking/gotrack_output"
            / str(task["object_name"]) / "world_pose_records.json"
        )
        pose = _pose_at(records_path, selected_frame)
        target = episode / args.output_name
        if target.exists() and not args.overwrite:
            existing.append(target)
        poses[episode_rel] = pose
        records.append({
            "episode_rel": episode_rel,
            "task_id": task["task_id"],
            "object": task["object_name"],
            "source_object": task.get("source_object"),
            "episode": task.get("episode"),
            "schedule_id": schedule_id,
            "attempt_dir": task["attempt_dir"],
            "init_frame_index": task.get("init_frame_index"),
            "selected_frame": selected_frame,
            "selected_timestamp": selection.get("selected_timestamp"),
            "human_episode": selection.get("human_episode"),
            "robot_episode": selection.get("robot_episode"),
            "selection_rel": str(selection_path.relative_to(shared)),
            "selection_sha256": _sha256(selection_path),
            "records_rel": str(records_path.relative_to(shared)),
            "records_sha256": _sha256(records_path),
            "target_rel": str(target.relative_to(shared)),
        })

    counts: dict[str, int] = {}
    for record in records:
        name = str(record["object"])
        counts[name] = counts.get(name, 0) + 1
    print(f"tasks={len(records)} by_object={counts}")
    if existing:
        raise FileExistsError(
            f"{len(existing)} targets exist; pass --overwrite only after review. First: {existing[0]}"
        )
    manifest_path = shared / args.manifest_rel
    if args.write and manifest_path.exists() and not args.overwrite:
        raise FileExistsError(manifest_path)
    if not args.write:
        for record in records:
            print(
                f"[audit] {record['target_rel']} <- {record['records_rel']} "
                f"frame={record['selected_frame']}"
            )
        return 0

    for record in records:
        target = shared / str(record["target_rel"])
        _atomic_npz(target, poses[str(record["episode_rel"])])
        record["target_sha256"] = _sha256(target)
        print(f"[wrote] {target}")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schedule_ids_oldest_to_newest": args.schedule_id,
        "target_root_rel": args.target_root_rel,
        "output_name": args.output_name,
        "tasks": len(records),
        "by_object": counts,
        "records": records,
    }
    _atomic_json(manifest_path, manifest)
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
