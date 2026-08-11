#!/usr/bin/env python3
"""Promote latest static FoundPose results into capture-local pose archives.

Schedules are ordered oldest-to-newest; a later completed task supersedes an
earlier one.  Rank zero uses the pipeline's published ``init_pose_world.npy``.
Explicit nonzero ranks are read from the refined ``candidate_bank.json``.
The default is a read-only audit.  ``--write`` atomically creates the archives
and a central provenance manifest.
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


def _latest_completed(schedules: list[Path]) -> dict[str, tuple[str, dict[str, Any]]]:
    latest: dict[str, tuple[str, dict[str, Any]]] = {}
    for schedule in schedules:
        for path in sorted((schedule / "tasks").glob("*.json")):
            task = _read_json(path)
            if (
                task.get("status") == "completed"
                and task.get("attempt_dir")
                and task.get("episode_rel")
            ):
                latest[str(task["episode_rel"])] = (schedule.name, task)
    return latest


def _resolve_overrides(
    latest: dict[str, tuple[str, dict[str, Any]]], raw: dict[str, int],
) -> dict[str, int]:
    """Resolve overrides by episode path, or by an unambiguous legacy task ID."""
    by_task_id: dict[str, list[str]] = {}
    for episode_rel, (_, task) in latest.items():
        by_task_id.setdefault(str(task["task_id"]), []).append(episode_rel)
    resolved: dict[str, int] = {}
    unknown: list[str] = []
    for selector, rank in raw.items():
        if selector in latest:
            resolved[selector] = rank
            continue
        matches = by_task_id.get(selector, [])
        if len(matches) == 1:
            resolved[matches[0]] = rank
        elif len(matches) > 1:
            raise KeyError(
                f"Ambiguous rank override {selector!r}; use one of these episode paths: {matches}"
            )
        else:
            unknown.append(selector)
    if unknown:
        raise KeyError(f"Rank overrides do not match completed tasks: {sorted(unknown)}")
    return resolved


def _snapshot_provenance(shared: Path, episode: Path) -> dict[str, Any] | None:
    snapshot_dir = episode / "grasp_snapshot"
    selection_path = snapshot_dir / "selection.json"
    robot_state_path = snapshot_dir / "robot_state/robot_state.npz"
    robot_metadata_path = snapshot_dir / "robot_state/metadata.json"
    if not any(path.is_file() for path in (selection_path, robot_state_path, robot_metadata_path)):
        return None
    result: dict[str, Any] = {}
    if selection_path.is_file():
        selection = _read_json(selection_path)
        result.update({
            "selection_rel": str(selection_path.relative_to(shared)),
            "selection_sha256": _sha256(selection_path),
            "selected_frame": selection.get("selected_frame"),
            "selected_timestamp": selection.get("selected_timestamp"),
            "human_episode": selection.get("human_episode"),
            "robot_episode": selection.get("robot_episode"),
        })
    if robot_state_path.is_file():
        result.update({
            "robot_state_rel": str(robot_state_path.relative_to(shared)),
            "robot_state_sha256": _sha256(robot_state_path),
        })
    if robot_metadata_path.is_file():
        metadata = _read_json(robot_metadata_path)
        result.update({
            "robot_metadata_rel": str(robot_metadata_path.relative_to(shared)),
            "robot_metadata_sha256": _sha256(robot_metadata_path),
            "arm_available": metadata.get("arm_available"),
        })
    images_dir = episode / "raw/images"
    if images_dir.is_dir():
        result["images_rel"] = str(images_dir.relative_to(shared))
        result["image_count"] = len(list(images_dir.glob("*.png")))
    return result


def _pose_for(shared: Path, task: dict[str, Any], rank: int) -> tuple[np.ndarray, Path, str]:
    init_dir = shared / task["attempt_dir"] / "foundpose_init"
    if rank == 0:
        source = init_dir / "init_pose_world.npy"
        pose = np.load(source)
        source_label = "published_init_pose_world"
    else:
        source = init_dir / "candidate_bank.json"
        candidates = _read_json(source).get("candidates", [])
        if rank >= len(candidates):
            raise IndexError(
                f"Requested rank {rank}, but {source} contains {len(candidates)} candidates"
            )
        pose = np.asarray(candidates[rank]["pose_world"], dtype=np.float64)
        source_label = f"refined_candidate_rank_{rank}"
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"Invalid pose for {task['task_id']}: {pose.shape}")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"Invalid homogeneous row for {task['task_id']}")
    return pose, source, source_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-id", nargs="+", required=True)
    parser.add_argument(
        "--runs-root-rel",
        default="object_tracking/campaigns/corl_rebuttal/foundpose_static_runs",
    )
    parser.add_argument("--shared-root-rel", default="shared_data")
    parser.add_argument("--target-root-rel", default="capture/corl_rebuttal")
    parser.add_argument("--rank-overrides-json", default=None)
    parser.add_argument("--output-name", default="object_6d_pose_v2.npz")
    parser.add_argument(
        "--manifest-rel",
        default="object_tracking/campaigns/corl_rebuttal/promotions/object_6d_pose_v2_manifest.json",
    )
    parser.add_argument("--expected-tasks", type=int, default=None)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shared = Path.home() / args.shared_root_rel
    runs_root = shared / args.runs_root_rel
    target_root = (shared / args.target_root_rel).resolve()
    schedules = [runs_root / value for value in args.schedule_id]
    missing = [str(path) for path in schedules if not (path / "manifest.json").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing schedules: {missing}")
    overrides: dict[str, int] = {}
    if args.rank_overrides_json:
        payload = _read_json(Path(args.rank_overrides_json).expanduser().resolve())
        if not isinstance(payload, dict):
            raise ValueError("Rank overrides must be a JSON object")
        raw_overrides = {str(key): int(value) for key, value in payload.items()}
        if any(rank < 0 for rank in raw_overrides.values()):
            raise ValueError("Candidate ranks must be non-negative")
    else:
        raw_overrides = {}

    latest = _latest_completed(schedules)
    if args.expected_tasks is not None and len(latest) != args.expected_tasks:
        raise RuntimeError(f"Expected {args.expected_tasks} tasks, found {len(latest)}")
    overrides = _resolve_overrides(latest, raw_overrides)

    records: list[dict[str, Any]] = []
    existing: list[str] = []
    for episode_rel, (schedule_id, task) in sorted(latest.items()):
        task_id = str(task["task_id"])
        episode = (shared / task["episode_rel"]).resolve()
        if episode != target_root and target_root not in episode.parents:
            raise ValueError(f"Task target is outside protected promotion root: {episode}")
        target = episode / args.output_name
        if target.exists() and not args.overwrite:
            existing.append(str(target))
        rank = overrides.get(episode_rel, 0)
        pose, source, source_label = _pose_for(shared, task, rank)
        record = {
            "episode_rel": episode_rel,
            "task_id": task_id,
            "object": task.get("source_object", task.get("object_name")),
            "mesh_object": task.get("object_name"),
            "episode": task.get("episode"),
            "schedule_id": schedule_id,
            "attempt_dir": task["attempt_dir"],
            "candidate_rank": rank,
            "pose_source": source_label,
            "source_rel": str(source.relative_to(shared)),
            "source_sha256": _sha256(source),
            "target_rel": str(target.relative_to(shared)),
            "pose": pose,
        }
        snapshot = _snapshot_provenance(shared, episode)
        if snapshot is not None:
            record["grasp_snapshot"] = snapshot
        records.append(record)

    source_counts: dict[str, int] = {}
    mesh_counts: dict[str, int] = {}
    for record in records:
        source = str(record["object"])
        mesh_object = str(record["mesh_object"])
        source_counts[source] = source_counts.get(source, 0) + 1
        mesh_counts[mesh_object] = mesh_counts.get(mesh_object, 0) + 1
    print(
        f"tasks={len(records)} by_object={mesh_counts} "
        f"by_source_object={source_counts} overrides={overrides}"
    )
    if existing:
        raise FileExistsError(
            f"{len(existing)} targets already exist; pass --overwrite only after review. "
            f"First: {existing[0]}"
        )
    manifest_path = shared / args.manifest_rel
    if args.write and manifest_path.exists() and not args.overwrite:
        raise FileExistsError(manifest_path)
    if not args.write:
        for record in records:
            print(
                f"[audit] {record['target_rel']} <- {record['schedule_id']} "
                f"rank={record['candidate_rank']}"
            )
        return 0

    manifest_records: list[dict[str, Any]] = []
    for record in records:
        target = shared / record["target_rel"]
        _atomic_npz(target, record.pop("pose"))
        record["target_sha256"] = _sha256(target)
        manifest_records.append(record)
        print(f"[wrote] {target}")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schedule_ids_oldest_to_newest": args.schedule_id,
        "target_root_rel": args.target_root_rel,
        "output_name": args.output_name,
        "rank_overrides": overrides,
        "tasks": len(manifest_records),
        "by_object": mesh_counts,
        "by_source_object": source_counts,
        "records": manifest_records,
    }
    _atomic_json(manifest_path, manifest)
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
