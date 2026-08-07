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
            if task.get("status") == "completed" and task.get("attempt_dir"):
                latest[str(task["task_id"])] = (schedule.name, task)
    return latest


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
        overrides = {str(key): int(value) for key, value in payload.items()}
        if any(rank < 0 for rank in overrides.values()):
            raise ValueError("Candidate ranks must be non-negative")

    latest = _latest_completed(schedules)
    if args.expected_tasks is not None and len(latest) != args.expected_tasks:
        raise RuntimeError(f"Expected {args.expected_tasks} tasks, found {len(latest)}")
    unknown = sorted(set(overrides) - set(latest))
    if unknown:
        raise KeyError(f"Rank overrides do not match completed tasks: {unknown}")

    records: list[dict[str, Any]] = []
    existing: list[str] = []
    for task_id, (schedule_id, task) in sorted(latest.items()):
        episode = (shared / task["episode_rel"]).resolve()
        if episode != target_root and target_root not in episode.parents:
            raise ValueError(f"Task target is outside protected promotion root: {episode}")
        target = episode / args.output_name
        if target.exists() and not args.overwrite:
            existing.append(str(target))
        rank = overrides.get(task_id, 0)
        pose, source, source_label = _pose_for(shared, task, rank)
        records.append({
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
        })

    counts: dict[str, int] = {}
    for record in records:
        counts[record["object"]] = counts.get(record["object"], 0) + 1
    print(f"tasks={len(records)} by_object={counts} overrides={overrides}")
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
        "by_object": counts,
        "records": manifest_records,
    }
    _atomic_json(manifest_path, manifest)
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
