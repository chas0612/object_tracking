#!/usr/bin/env python3
"""Sequential, restart-safe FoundPose + GoTrack runner for the hand-over captures.

Runs these jobs in order:
  1. capture 1 / circular_frying_pan
  2. capture 2 / circular_frying_pan (reuses capture 1's FoundPose repre)
  3. capture 2 / orange

Completed stages are skipped by checking their final output files.  Existing
partial directories are never removed or overwritten: the script stops and
prints the affected path so a new explicit output name can be selected.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_ROOT = Path("/home/capture16/shared_data/capture/hand_over/seongho/bimanual/circular_frying_pan")
MESH_ROOT = Path("/home/capture16/shared_data/mesh_blender")
# Use 12 first.  If this GPU runs out of memory, change only this value to 8
# and rerun; completed mask/FoundPose stages will be skipped automatically.
NUM_CAMERAS = 12


@dataclass(frozen=True)
class Job:
    capture: Path
    object_name: str
    mesh: Path
    tag: str

    @property
    def frame_dir(self) -> Path:
        return self.capture / f"foundpose_frame_000000_{self.tag}"

    @property
    def init_dir(self) -> Path:
        return self.capture / f"foundpose_{self.tag}"

    def tracking_dir(self, cameras: int) -> Path:
        return self.capture / f"gotrack_{self.tag}_{cameras}cam"


def _is_complete_mask(job: Job, video_dir: Path) -> bool:
    metadata_path = job.frame_dir / "metadata.json"
    if not metadata_path.is_file() or not (job.frame_dir / "images").is_dir() or not (job.frame_dir / "masks").is_dir():
        return False
    try:
        metadata = __import__("json").loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return metadata.get("source_video_dir") == str(video_dir.resolve())


def _is_complete_init(job: Job) -> bool:
    return (job.init_dir / "init_pose_world.npy").is_file()


def _is_complete_track(job: Job, cameras: int) -> bool:
    output = job.tracking_dir(cameras) / "gotrack_output" / job.object_name
    return (output / "world_pose_records.json").is_file() and (output / "summary.json").is_file()


def _is_complete_undistort(capture: Path) -> bool:
    source = {p.name for p in (capture / "videos").glob("*.avi")}
    generated = capture / "undistorted_video"
    manifest_path = generated / "undistort_manifest.json"
    if not source or not generated.is_dir() or not manifest_path.is_file():
        return False
    try:
        manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if manifest.get("new_intrinsics_field") != "intrinsics_undistort":
        return False
    return all(
        (generated / name).is_file() and (generated / name).stat().st_size > 0 for name in source
    )


def _check_partial(path: Path, complete: bool, label: str) -> None:
    if path.exists() and not complete:
        raise RuntimeError(
            f"Existing incomplete {label} output was found: {path}\n"
            "It is preserved and will not be overwritten. Inspect it or use a new explicit output directory."
        )


def _run(command: list[str], dry_run: bool) -> None:
    print("[run]", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def run_job(job: Job, cameras: int, dry_run: bool) -> None:
    print(f"\n=== {job.capture.name} / {job.object_name} ===", flush=True)
    if not job.capture.is_dir() or not job.mesh.is_file():
        raise FileNotFoundError(f"Missing capture or mesh: {job.capture}, {job.mesh}")

    undistorted_dir = job.capture / "undistorted_video"
    if _is_complete_undistort(job.capture):
        print(f"[skip] undistort: {undistorted_dir}", flush=True)
    else:
        _run([
            "conda", "run", "--no-capture-output", "-n", "gotrack", "python", "-u",
            "src/process/undistort_capture_videos.py", "--capture-dir", str(job.capture),
        ], dry_run)

    mask_done = _is_complete_mask(job, undistorted_dir)
    _check_partial(job.frame_dir, mask_done, "mask")
    if mask_done:
        print(f"[skip] mask: {job.frame_dir}", flush=True)
    else:
        _run([
            "conda", "run", "--no-capture-output", "-n", "sam3", "python", "-u", "src/process/mask.py",
            "--capture_dir", str(job.capture), "--frame-index", "0", "--prompt", job.object_name.replace("_", " "),
            "--frame-output-dir", str(job.frame_dir), "--video-dir", str(undistorted_dir),
        ], dry_run)

    init_done = _is_complete_init(job)
    _check_partial(job.init_dir, init_done, "FoundPose init")
    if init_done:
        print(f"[skip] FoundPose init: {job.init_dir / 'init_pose_world.npy'}", flush=True)
    else:
        command = [
            "conda", "run", "--no-capture-output", "-n", "gotrack", "python", "-u",
            "src/process/foundpose_init_capture.py", "--capture-dir", str(job.capture),
            "--frame-dir", str(job.frame_dir), "--mesh", str(job.mesh),
            "--object-name", job.object_name, "--output-dir", str(job.init_dir),
        ]
        _run(command, dry_run)

    tracking_dir = job.tracking_dir(cameras)
    track_done = _is_complete_track(job, cameras)
    _check_partial(tracking_dir, track_done, "GoTrack")
    if track_done:
        print(f"[skip] GoTrack: {tracking_dir / 'gotrack_output' / job.object_name / 'world_pose_records.json'}", flush=True)
    else:
        _run([
            "conda", "run", "--no-capture-output", "-n", "gotrack", "python", "-u",
            "src/process/gotrack_capture.py", "--capture-dir", str(job.capture), "--mesh", str(job.mesh),
            "--init-pose", str(job.init_dir / "init_pose_world.npy"), "--object-name", job.object_name,
            "--video-dir", str(undistorted_dir), "--num-cameras", str(cameras), "--max-frames", "-1", "--output-dir", str(tracking_dir),
        ], dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-cameras", type=int, default=NUM_CAMERAS,
                        help=f"Default: NUM_CAMERAS={NUM_CAMERAS}; override for this run if needed.")
    parser.add_argument("--dry-run", action="store_true", help="Print only the stages that would run.")
    args = parser.parse_args()
    if args.num_cameras < 1:
        raise ValueError("--num-cameras must be positive")

    cap1, cap2 = CAPTURE_ROOT / "1", CAPTURE_ROOT / "2"
    pan_mesh = MESH_ROOT / "circular_frying_pan" / "circular_frying_pan.obj"
    orange_mesh = MESH_ROOT / "orange" / "orange.obj"
    pan1 = Job(cap1, "circular_frying_pan", pan_mesh, "pan")
    jobs = [
        pan1,
        # FoundPose assets now default to <mesh parent>/foundpose_assets, so
        # capture 1 and capture 2 automatically share the pan representation.
        Job(cap2, "circular_frying_pan", pan_mesh, "pan"),
        Job(cap2, "orange", orange_mesh, "orange"),
    ]
    for job in jobs:
        run_job(job, args.num_cameras, args.dry_run)
    print("\n[done] all requested jobs are complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
