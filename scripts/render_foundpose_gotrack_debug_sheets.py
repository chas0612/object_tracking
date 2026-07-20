#!/usr/bin/env python3
"""Render one sparse GoTrack reprojection sheet for every completed scheduler task.

Run this on one CUDA-capable machine in the ``gotrack`` environment.  It is
deliberately sequential: each task reads only a handful of NAS video frames,
renders one contact sheet, and writes one JPEG.  Existing sheets are skipped,
so a stopped batch can be resumed safely.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SHEET_RENDERER = REPO_ROOT / "src/process/render_gotrack_debug_sheet.py"


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-id", required=True)
    parser.add_argument("--shared-root-rel", default="shared_data")
    parser.add_argument("--output-root-rel", default="object_tracking/gotrack_debug_sheets",
                        help="Directory under ~/shared_data; sheets are stored in a schedule subdirectory.")
    parser.add_argument("--max-cameras", type=int, default=6)
    parser.add_argument("--frame-indices", nargs="*", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 means every completed task.")
    parser.add_argument("--force", action="store_true",
                        help="Replace existing episode and central sheets for selected completed tasks.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_cameras < 1 or args.limit < 0:
        raise ValueError("--max-cameras must be positive and --limit non-negative")
    if any(Path(value).is_absolute() or ".." in Path(value).parts
           for value in (args.shared_root_rel, args.output_root_rel)):
        raise ValueError("root paths must be safe relative paths")

    shared = Path.home() / args.shared_root_rel
    schedule = shared / "object_tracking/foundpose_gotrack_runs" / args.schedule_id
    tasks_dir = schedule / "tasks"
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"Task directory not found: {tasks_dir}")
    output_dir = shared / args.output_root_rel / args.schedule_id
    tasks: list[dict] = []
    for path in sorted(tasks_dir.glob("*.json")):
        task = _read_json(path)
        if task.get("status") == "completed":
            tasks.append(task)
    if args.limit:
        tasks = tasks[:args.limit]

    rendered = skipped = unavailable = failed = 0
    for number, task in enumerate(tasks, start=1):
        task_id = str(task["task_id"])
        episode = shared / task["episode_rel"]
        mesh = shared / task["mesh_rel"]
        attempt = shared / task["attempt_dir"]
        records = attempt / "gotrack_tracking/gotrack_output" / str(task["object_name"]) / "world_pose_records.json"
        output = output_dir / f"{task_id}.jpg"
        episode_output = episode / f"gotrack_debug_sheet_{args.schedule_id}_{attempt.name}.jpg"
        if args.force and not args.dry_run:
            # Only touch the two known output paths for this scheduler task.
            # This avoids recursively scanning every capture directory on NAS.
            for existing in (episode_output, output):
                if existing.is_symlink() or existing.is_file():
                    existing.unlink()
        if output.is_file() and episode_output.is_file():
            print(f"[{number}/{len(tasks)}] skip existing {task_id}", flush=True)
            skipped += 1
            continue
        if not (episode.is_dir() and mesh.is_file() and records.is_file()):
            print(f"[{number}/{len(tasks)}] unavailable {task_id}: episode={episode.is_dir()} mesh={mesh.is_file()} records={records.is_file()}", flush=True)
            unavailable += 1
            continue
        if episode_output.is_file():
            if args.dry_run:
                print(f"[{number}/{len(tasks)}] would copy episode sheet to central store: {task_id}", flush=True)
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(episode_output, output)
            print(f"[{number}/{len(tasks)}] copied episode sheet to central store: {task_id}", flush=True)
            skipped += 1
            continue
        if output.is_file():
            if args.dry_run:
                print(f"[{number}/{len(tasks)}] would copy central sheet to episode: {task_id}", flush=True)
                continue
            shutil.copy2(output, episode_output)
            print(f"[{number}/{len(tasks)}] copied central sheet to episode: {task_id}", flush=True)
            skipped += 1
            continue
        command = [sys.executable, "-u", str(SHEET_RENDERER),
                   "--capture-dir", str(episode), "--object-mesh", str(mesh),
                   "--gotrack-records", str(records), "--output", str(episode_output),
                   "--max-cameras", str(args.max_cameras)]
        if args.frame_indices:
            command.extend(["--frame-indices", *(str(frame) for frame in args.frame_indices)])
        print(f"[{number}/{len(tasks)}] {task_id}", flush=True)
        if args.dry_run:
            print("  " + " ".join(command), flush=True)
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(command, cwd=REPO_ROOT)
        if result.returncode == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(episode_output, output)
            rendered += 1
        else:
            failed += 1
            print(f"  FAILED returncode={result.returncode}", flush=True)
    print(f"[summary] completed_tasks={len(tasks)} rendered={rendered} skipped={skipped} unavailable={unavailable} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
