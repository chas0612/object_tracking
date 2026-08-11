#!/usr/bin/env python3
"""Render CORL grasp GoTrack trajectories through each selected grasp frame."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RENDERER = REPO_ROOT / "src/process/render_robot_object_reprojection_grid.py"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe(value: str) -> str:
    rendered = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return rendered.strip("._") or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-id", required=True)
    parser.add_argument(
        "--runs-root-rel",
        default="object_tracking/campaigns/corl_rebuttal/allegro_v5_grasp_gotrack_runs",
    )
    parser.add_argument("--shared-root-rel", default="shared_data")
    parser.add_argument(
        "--output-root-rel",
        default="object_tracking/campaigns/corl_rebuttal/allegro_v5_grasp_gotrack_grids",
    )
    parser.add_argument("--grid-scale", type=float, default=0.15)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shared = Path.home() / args.shared_root_rel
    schedule = shared / args.runs_root_rel / args.schedule_id
    if not (schedule / "manifest.json").is_file():
        raise FileNotFoundError(schedule)
    tasks = []
    for path in sorted((schedule / "tasks").glob("*.json")):
        task = _read_json(path)
        if task.get("status") == "completed" and task.get("attempt_dir"):
            tasks.append(task)
    output_root = shared / args.output_root_rel
    rendered = skipped = failed = 0
    for index, task in enumerate(tasks, start=1):
        episode = shared / str(task["episode_rel"])
        selection = _read_json(episode / "grasp_snapshot/selection.json")
        selected_frame = int(selection["selected_frame"])
        parts = Path(str(task["episode_rel"])).parts
        operator = parts[-3] if len(parts) >= 3 else "unknown"
        output = output_root / _safe(operator) / f"{_safe(str(task['task_id']))}.mp4"
        if output.exists() and not args.force:
            print(f"[{index}/{len(tasks)}] skip {output}", flush=True)
            skipped += 1
            continue
        if output.exists() and args.force and not args.dry_run:
            output.unlink()
        records = (
            shared / str(task["attempt_dir"]) / "gotrack_tracking/gotrack_output"
            / str(task["object_name"]) / "world_pose_records.json"
        )
        command = [
            sys.executable, "-u", str(RENDERER),
            "--capture-dir", str(episode),
            "--video-dir", str(episode / "undistorted_video"),
            "--object-mesh", str(shared / str(task["mesh_rel"])),
            "--gotrack-records", str(records),
            "--object-only",
            "--start-frame", str(args.start_frame),
            "--end-frame", str(selected_frame),
            "--grid-scale", str(args.grid_scale),
            "--output", str(output),
        ]
        print(
            f"[{index}/{len(tasks)}] {task['episode_rel']} frames="
            f"{args.start_frame}..{selected_frame}", flush=True,
        )
        if args.dry_run:
            print("  " + " ".join(command), flush=True)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(command, cwd=REPO_ROOT)
        if result.returncode == 0:
            rendered += 1
        else:
            failed += 1
            print(f"  FAILED returncode={result.returncode}", flush=True)
    print(
        f"[summary] tasks={len(tasks)} rendered={rendered} skipped={skipped} failed={failed}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
