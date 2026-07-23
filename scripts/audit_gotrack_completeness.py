#!/usr/bin/env python3
"""Audit scheduler-completed GoTrack records against capture timestamps.

Older schedules treated a process exit as success even when tracking later
lost all poses.  This utility reports those silent incomplete outputs without
changing scheduler state or capture results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _expected_frames(episode: Path) -> tuple[int | None, str]:
    timestamps = episode / "raw" / "timestamps" / "timestamp.npy"
    if timestamps.is_file():
        return len(np.load(timestamps, mmap_mode="r")), "timestamps"
    object_pose_dir = episode / "object_6d"
    if object_pose_dir.is_dir():
        files = sorted(object_pose_dir.glob("pose_*.txt"))
        if files:
            return len(files), "object_6d"
    return None, "unavailable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-id", required=False)
    parser.add_argument("--all-schedules", action="store_true",
                        help="Inspect every scheduler run under shared_data/object_tracking.")
    parser.add_argument("--shared-root-rel", default="shared_data")
    parser.add_argument("--runs-root-rel", default="object_tracking/foundpose_gotrack_runs")
    parser.add_argument("--min-valid-pose-coverage", type=float, default=0.5)
    parser.add_argument("--max-trailing-missing-frames", type=int, default=30)
    parser.add_argument("--all-statuses", action="store_true",
                        help="Inspect every task with records, rather than only scheduler-completed tasks.")
    parser.add_argument("--only-incomplete", action="store_true",
                        help="Suppress harmless record/timestamp count mismatch-only rows.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.all_schedules and not args.schedule_id:
        raise ValueError("Pass --schedule-id or --all-schedules")
    shared = Path.home() / args.shared_root_rel
    if Path(args.runs_root_rel).is_absolute() or ".." in Path(args.runs_root_rel).parts:
        raise ValueError("--runs-root-rel must be a safe relative path")
    runs_root = shared / args.runs_root_rel
    tasks_dirs = (sorted(path / "tasks" for path in runs_root.iterdir() if (path / "tasks").is_dir())
                  if args.all_schedules else [runs_root / str(args.schedule_id) / "tasks"])
    if not tasks_dirs:
        raise FileNotFoundError(f"No scheduler task directories under: {runs_root}")
    for tasks_dir in tasks_dirs:
        if not tasks_dir.is_dir():
            raise FileNotFoundError(f"Task directory not found: {tasks_dir}")

    checked = missing = incomplete = frame_mismatch = 0
    for tasks_dir in tasks_dirs:
        schedule_id = tasks_dir.parent.name
        for task_path in sorted(tasks_dir.glob("*.json")):
            task = json.loads(task_path.read_text(encoding="utf-8"))
            if not args.all_statuses and task.get("status") != "completed":
                continue
            episode = shared / task["episode_rel"]
            attempt_rel = task.get("attempt_dir")
            if not attempt_rel:
                missing += 1
                continue
            records_path = (shared / attempt_rel / "gotrack_tracking" / "gotrack_output" /
                            str(task["object_name"]) / "world_pose_records.json")
            expected_frames, frame_source = _expected_frames(episode)
            if not records_path.is_file() or expected_frames is None:
                missing += 1
                continue

            records = json.loads(records_path.read_text(encoding="utf-8"))
            valid = sum(isinstance(row, dict) and row.get("pose_world") is not None for row in records)
            trailing_missing = 0
            for row in reversed(records):
                if isinstance(row, dict) and row.get("pose_world") is not None:
                    break
                trailing_missing += 1
            coverage = valid / len(records) if records else 0.0
            is_mismatch = len(records) != expected_frames
            is_incomplete = (coverage < args.min_valid_pose_coverage or
                             trailing_missing > args.max_trailing_missing_frames)
            checked += 1
            frame_mismatch += int(is_mismatch)
            incomplete += int(is_incomplete)
            if is_incomplete or (is_mismatch and not args.only_incomplete):
                flags = []
                if is_incomplete:
                    flags.append("incomplete")
                if is_mismatch:
                    flags.append("frame_mismatch")
                print(
                    f"[{','.join(flags)}] {schedule_id}/{task['task_id']} "
                    f"expected={expected_frames} source={frame_source} records={len(records)} "
                    f"valid={valid}/{len(records)} ({coverage:.1%}) trailing_missing={trailing_missing}",
                    flush=True,
                )
    print(f"[summary] checked={checked} incomplete={incomplete} "
          f"frame_mismatch={frame_mismatch} missing={missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
