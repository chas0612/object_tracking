#!/usr/bin/env python3
"""Render latest successful static-FoundPose tasks into one QA directory.

Schedules are supplied oldest-to-newest.  When the same task appears more than
once for the same capture episode, the newest completed result wins.  Capture
episode paths, rather than task IDs, are the identity because task IDs can be
reused by independent operators.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _completed_tasks(schedule: Path) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for path in sorted((schedule / "tasks").glob("*.json")):
        task = _read_json(path)
        if task.get("status") != "completed":
            continue
        required = ("task_id", "episode_rel", "mesh_rel", "attempt_dir")
        if all(task.get(key) for key in required):
            selected[str(task["episode_rel"])] = task
    return selected


def _safe_component(value: str) -> str:
    rendered = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return rendered.strip("._") or "unknown"


def _output_relative(task: dict[str, Any]) -> Path:
    parts = Path(str(task["episode_rel"])).parts
    operator = parts[-3] if len(parts) >= 3 else "unknown"
    return Path(_safe_component(operator)) / f"{_safe_component(str(task['task_id']))}.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-id", nargs="+", required=True,
                        help="Schedule IDs in oldest-to-newest priority order.")
    parser.add_argument(
        "--runs-root-rel",
        default="object_tracking/campaigns/corl_rebuttal/foundpose_static_runs",
    )
    parser.add_argument("--shared-root-rel", default="shared_data")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-cameras", type=int, default=22)
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--cell-width", type=int, default=320)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_cameras < 1 or args.columns < 1 or args.cell_width < 64:
        raise ValueError("invalid grid dimensions")
    shared = Path.home() / args.shared_root_rel
    runs_root = shared / args.runs_root_rel
    schedules = [runs_root / schedule_id for schedule_id in args.schedule_id]
    missing = [str(path) for path in schedules if not (path / "manifest.json").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing schedules: {missing}")

    latest: dict[str, dict[str, Any]] = {}
    for schedule in schedules:
        latest.update(_completed_tasks(schedule))
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else runs_root / f"{args.schedule_id[-1]}_debug_sheets"
    )
    if args.dry_run:
        print(f"selected_completed_tasks={len(latest)}")
        for episode_rel, task in sorted(latest.items()):
            print(f"{episode_rel} -> {_output_relative(task)}")
        print(f"output={output_dir}")
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered = skipped = failed = 0
    failures: dict[str, str] = {}
    for index, (episode_rel, task) in enumerate(sorted(latest.items()), start=1):
        task_id = str(task["task_id"])
        output = output_dir / _output_relative(task)
        if output.exists() and not args.overwrite:
            skipped += 1
            continue
        if output.exists():
            output.unlink()
        attempt = shared / task["attempt_dir"]
        frame_dirs = sorted(attempt.glob("foundpose_frame_*"))
        if len(frame_dirs) != 1:
            failures[episode_rel] = f"expected one foundpose_frame_* directory, got {len(frame_dirs)}"
            failed += 1
            continue
        command = [
            sys.executable, str(REPO_ROOT / "scripts/render_foundpose_init_debug_sheet.py"),
            "--capture-dir", str(shared / task["episode_rel"]),
            "--object-mesh", str(shared / task["mesh_rel"]),
            "--foundpose-frame-dir", str(frame_dirs[0]),
            "--init-pose", str(attempt / "foundpose_init/init_pose_world.npy"),
            "--output", str(output),
            "--max-cameras", str(args.max_cameras),
            "--columns", str(args.columns),
            "--cell-width", str(args.cell_width),
            "--include-unmasked",
        ]
        print(f"[{index}/{len(latest)}] {episode_rel}", flush=True)
        result = subprocess.run(command, cwd=REPO_ROOT)
        if result.returncode == 0:
            rendered += 1
        else:
            failed += 1
            failures[episode_rel] = f"renderer_returncode={result.returncode}"

    summary = {
        "schedule_ids": args.schedule_id,
        "selected_completed_tasks": len(latest),
        "rendered": rendered,
        "skipped_existing": skipped,
        "failed": failed,
        "failures": failures,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"output={output_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
