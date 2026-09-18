#!/usr/bin/env python3
"""Render compact pose-overlay grid videos from one or more phased schedules."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RENDERER = REPO_ROOT / "src/process/render_robot_object_reprojection_grid.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.process.gotrack_capture import _video_paths_by_stem


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _evenly_spaced(values: list[str], count: int) -> list[str]:
    if len(values) <= count:
        return values
    indices = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[int(index)] for index in indices]


def _select_cameras(video_dir: Path, count: int, max_skew_sec: float) -> list[str]:
    timings: list[tuple[str, int, float]] = []
    for path in _video_paths_by_stem(video_dir).values():
        cap = cv2.VideoCapture(str(path))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        if frames >= 100 and fps > 0:
            timings.append((path.stem, frames, frames / fps))
    if not timings:
        raise RuntimeError(f"No usable videos in {video_dir}")
    median = float(np.median([item[2] for item in timings]))
    usable = [serial for serial, _, duration in timings
              if abs(duration - median) <= max_skew_sec]
    return _evenly_spaced(usable, count)


def _records_for(shared: Path, task: dict, schedule_id: str) -> Path | None:
    episode = shared / task["episode_rel"]
    base = episode / "object_tracking_foundpose_gotrack" / schedule_id
    matches = sorted(base.glob(
        f"attempt_*/gotrack_tracking/gotrack_output/{task['object_name']}/world_pose_records.json"
    ))
    return matches[-1] if matches else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-id", action="append", required=True)
    parser.add_argument("--runs-root-rel", required=True)
    parser.add_argument("--output-root-rel", required=True)
    parser.add_argument("--shared-root-rel", default="shared_data")
    parser.add_argument("--max-cameras", type=int, default=4)
    parser.add_argument("--grid-scale", type=float, default=0.15)
    parser.add_argument("--max-video-duration-skew-sec", type=float, default=1.0)
    parser.add_argument("--object-only", action="store_true")
    parser.add_argument("--robot-profile", choices=("auto", "inspire", "robotiq_2f85"), default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_cameras < 1:
        raise ValueError("--max-cameras must be positive")
    shared = Path.home() / args.shared_root_rel
    runs = shared / args.runs_root_rel
    output_root = shared / args.output_root_rel

    # A later schedule wins if the same episode appears more than once.
    selected: dict[tuple[str, str], tuple[dict, str]] = {}
    for schedule_id in args.schedule_id:
        schedule = runs / schedule_id
        if not (schedule / "manifest.json").is_file():
            raise FileNotFoundError(schedule)
        for path in sorted((schedule / "tasks").glob("*.json")):
            task = _read(path)
            if task.get("phases", {}).get("gotrack", {}).get("status") == "completed":
                selected[(str(task["object_name"]), str(task["episode"]))] = (task, schedule_id)

    rendered = skipped = failed = 0
    print(f"[batch] selected={len(selected)} schedules={args.schedule_id}", flush=True)
    for index, ((object_name, episode_name), (task, schedule_id)) in enumerate(
        sorted(selected.items()), start=1
    ):
        episode = shared / task["episode_rel"]
        records = _records_for(shared, task, schedule_id)
        output = output_root / object_name / f"{episode_name}.mp4"
        if records is None:
            print(f"[{index}/{len(selected)}] FAIL {object_name}/{episode_name}: records missing", flush=True)
            failed += 1
            continue
        if output.exists() and not args.force:
            print(f"[{index}/{len(selected)}] skip {object_name}/{episode_name}", flush=True)
            skipped += 1
            continue
        temporary_output = output.with_name(output.stem + ".rendering.mp4")
        try:
            cameras = _select_cameras(
                episode / "videos", args.max_cameras, args.max_video_duration_skew_sec
            )
            command = [
                sys.executable, "-u", str(RENDERER),
                "--capture-dir", str(episode),
                "--video-dir", str(episode / "videos"),
                "--object-mesh", str(shared / task["mesh_rel"]),
                "--gotrack-records", str(records),
                "--robot-profile", args.robot_profile,
                "--camera-ids", *cameras,
                "--grid-scale", str(args.grid_scale),
                "--max-video-duration-skew-sec", str(args.max_video_duration_skew_sec),
                "--undistort-input",
                "--output", str(temporary_output),
            ]
            if args.object_only:
                command.append("--object-only")
            print(
                f"[{index}/{len(selected)}] render {object_name}/{episode_name} "
                f"schedule={schedule_id} cameras={','.join(cameras)}",
                flush=True,
            )
            if args.dry_run:
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            temporary_output.replace(output)
            rendered += 1
        except Exception as exc:
            if not args.dry_run:
                temporary_output.unlink(missing_ok=True)
            failed += 1
            print(f"[{index}/{len(selected)}] FAIL {object_name}/{episode_name}: {exc}", flush=True)
    print(f"[summary] total={len(selected)} rendered={rendered} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
